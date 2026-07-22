/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "healpix_util.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

static healpix_util::Healpix_Nested_CUDA* cached_healpix_gp = nullptr;
static size_t cached_healpix_elems = 0;
static int2* cached_d_stk_data = nullptr;
static size_t cached_stk_elems = 0;
static int* cached_tilesKey_touched = nullptr;
static size_t cached_tilesKey_elems = 0;

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__global__ void duplicateWithKeysOmni(
	int P,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	float* radii,
	int* tilesKey_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	if (radii[idx] > -1e-6)
	{
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		for (int t = off ; t < offsets[idx]; t++)
		{
			uint64_t key = tilesKey_touched[t];
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			gaussian_keys_unsorted[t] = key;
			gaussian_values_unsorted[t] = idx;
		}
	}
}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
			// if (prevtile <= 2){
			// 	printf("prevtile: %d, start: %d, end: %d\n", prevtile, ranges[prevtile].x, ranges[prevtile].y);
			// }
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	obtain(chunk, geom.means_xyf, P, 128);
	obtain(chunk, geom.means_nest, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	// obtain(chunk, binning.tilesKey_touched, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

healpix_util::HealpixState healpix_util::HealpixState::fromChunk(char*& chunk, int P, int stack_elements_per_thread, int omax){
	healpix_util::HealpixState Healpix;
	healpix_util::obtain(chunk, Healpix.healpix_gp, P * (omax + 1), 128);
	healpix_util::obtain(chunk, Healpix.d_stk_data, P * stack_elements_per_thread, 128);
	return Healpix;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}


int CudaRasterizer::OmniRasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> healpixBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const int N_side,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	float* hp_color,
	float* hp_depth,
	float* original_image,
	float* original_hp,
	int* radii,
	float* radius_rad,
	bool debug,
	bool skip_2d_output,
	bool use_quadtree_query)
{

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
		// radii = geomState.cov3D; // TODO: just a placeholder to avoid crash
	}

	// dim3 is the CUDA built-in 3D type
	// each block is BLOCK_X x BLOCK_Y threads, each tile is under a block for processing
	// Here we use the 3d tile grid to cover the whole sphere
	dim3 tile_grid((N_side + BLOCK_X_3D - 1) / BLOCK_X_3D, (N_side + BLOCK_Y_3D - 1) / BLOCK_Y_3D, 12);
	dim3 block(BLOCK_X_3D, BLOCK_Y_3D, 1);

	// First, convert original equirectangular image to healpix map for gt generation
	int N_pix = N_side * N_side * 12;
	int N_order = round(log2(N_side));
	// query_equi_from_hp<<<(N_pix+ 255) / 256, 256>>>(
	// 	width, height, N_order,
	// 	original_image,
	// 	original_hp,
	// 	nullptr,
	// 	nullptr
	// );


	// Dynamically resize image-based auxiliary buffers during training
	//Now consider the healpix as the image, with 12*N_side*N_side pixels
	
	size_t img_chunk_size = required<ImageState>(N_pix); //here the image id the healpix map
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, N_pix);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}
	
	
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocessOmni(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		N_side,
		radii,
		radius_rad,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		geomState.means_xyf,
		geomState.means_nest,
		prefiltered,
		block,
		antialiasing
	), debug)
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// std::cout << "Computing prefix sum of tiles touched..." << std::endl;
	// Given a sequence of input elements and a binary reduction operator, 
	// a prefix scan produces an output sequence where each element is computed to be the reduction of the elements occurring earlier in the input sequence.

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	// std::cout << "Retrieving number of rendered Gaussians..." << std::endl;
	int num_rendered; //host varaible
	const int fact = 2;  // Use inclusive boundary to avoid tile edge artifacts
	int order = round(log2(tile_grid.x));
	
	// Allocate quadtree buffers only if using quadtree method
	healpix_util::Healpix_Nested_CUDA* healpix_gp = nullptr;
	int2* d_stk_data = nullptr;
	int omax = 0;
	
	if (use_quadtree_query) {
		// Quadtree method (slower but more accurate)
		int oplus = (fact == 0) ? 0 : round(log2(fact));
		omax = order + oplus;
		const int order_max = 13;
		if (order + oplus > order_max) {
			printf("Error: downsampling factor too large for given nside! order=%d oplus=%d order_max=%d\n",
					order, oplus, order_max);
		}
		int stack_elements_per_thread = (12 + 6 * omax);
		size_t stk_elems = (size_t)P * (size_t)stack_elements_per_thread;
		size_t healpix_elems = (size_t)P * (size_t)(omax + 1);

		if (cached_healpix_elems < healpix_elems) {
			if (cached_healpix_gp) { cudaFree(cached_healpix_gp); cached_healpix_gp = nullptr; cached_healpix_elems = 0; }
			cudaError_t e = cudaMalloc(&cached_healpix_gp, healpix_elems * sizeof(healpix_util::Healpix_Nested_CUDA));
			if (e != cudaSuccess) { std::cerr << "[ALLOC ERROR] healpix_gp malloc failed: " << cudaGetErrorString(e) << std::endl; throw std::runtime_error("alloc healpix_gp"); }
			cudaMemset(cached_healpix_gp, 0, healpix_elems * sizeof(healpix_util::Healpix_Nested_CUDA));
			cached_healpix_elems = healpix_elems;
		}

		if (cached_stk_elems < stk_elems) {
			if (cached_d_stk_data) { cudaFree(cached_d_stk_data); cached_d_stk_data = nullptr; cached_stk_elems = 0; }
			cudaError_t e = cudaMalloc(&cached_d_stk_data, stk_elems * sizeof(int2));
			if (e != cudaSuccess) { std::cerr << "[ALLOC ERROR] d_stk_data malloc failed: " << cudaGetErrorString(e) << std::endl; throw std::runtime_error("alloc d_stk_data"); }
			cudaMemset(cached_d_stk_data, 0, stk_elems * sizeof(int2));
			cached_stk_elems = stk_elems;
		}
		healpix_gp = cached_healpix_gp;
		d_stk_data = cached_d_stk_data;
		
		// First query: count tiles touched per Gaussian
		query_disc_downsample<<<(P + 255) / 256, 256>>>(P, geomState.means2D, radius_rad, fact, tile_grid.x, healpix_gp, geomState.tiles_touched, d_stk_data);
	} else {
		// Fast rectangle bounding box method (default)
		query_disc_rect<<<(P + 255) / 256, 256>>>(P, geomState.means2D, radius_rad, fact, tile_grid.x, geomState.tiles_touched, nullptr, nullptr);
	}
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	// get the last element of point_offsets from device to host
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
	// size_t healpix_chunk_size = healpix_util::required<healpix_util::HealpixState>(num_rendered, P, stack_elements_per_thread, omax);
	// char* healpix_chunkptr = healpixBuffer(healpix_chunk_size);
	// healpix_util::HealpixState healpixState = healpix_util::HealpixState::fromChunk(healpix_chunkptr, num_rendered, P, stack_elements_per_thread, omax);
    size_t tilesKey_elems = (size_t)num_rendered;
	// printf("the size of the binning chunk size: %zu\n", binning_chunk_size);
	// printf("the size of the number of rendered gaussians: %d\n", num_rendered);
    // ensure previous kernels finished
    // cudaDeviceSynchronize();
    if (tilesKey_elems > 0 && cached_tilesKey_elems < tilesKey_elems) {
        if (cached_tilesKey_touched) { cudaFree(cached_tilesKey_touched); cached_tilesKey_touched = nullptr; cached_tilesKey_elems = 0; }
        cudaError_t e = cudaMalloc(&cached_tilesKey_touched, tilesKey_elems * sizeof(int));
        if (e != cudaSuccess) {
            std::cerr << "[ALLOC WARNING] tilesKey_touched malloc failed: " << cudaGetErrorString(e) << " (continuing with nullptr)\n";
            cached_tilesKey_touched = nullptr;
            cached_tilesKey_elems = 0;
        } else {
            cudaMemset(cached_tilesKey_touched, 0, tilesKey_elems * sizeof(int));
            cached_tilesKey_elems = tilesKey_elems;
        }
    } else if (tilesKey_elems == 0) {
        // nothing to allocate, ensure pointer null
        // cached_tilesKey_touched may remain from earlier frames; set to nullptr if you prefer.
    }
    int* tilesKey_touched = cached_tilesKey_touched;
	// printf("P: %d, num_rendered: %d, omax: %d, stack_elements_per_thread: %d\n", P,  num_rendered, omax, stack_elements_per_thread);
	// std::cout << "Querying disc downsample..." << std::endl;
	
	// Second query call: populate tilesKey_touched with actual tile indices
	if (use_quadtree_query) {
		// Quadtree method (slower but more accurate)
		query_disc_downsample_skip_init<<<(P + 255) / 256, 256>>>(
			P,
			geomState.means2D,
			radius_rad,
			fact,
			tile_grid.x,
			healpix_gp,
			geomState.tiles_touched,
			d_stk_data,
			tilesKey_touched,
			geomState.point_offsets
		);
	} else {
		// Fast rectangle bounding box method (default)
		query_disc_rect<<<(P + 255) / 256, 256>>>(
			P,
			geomState.means2D,
			radius_rad,
			fact,
			tile_grid.x,
			geomState.tiles_touched,
			tilesKey_touched,
			geomState.point_offsets
		);
	}

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// std::cout << "Generating keys and values for " << num_rendered << " rendered Gaussians..." << std::endl;
	duplicateWithKeysOmni << <(P + 255) / 256, 256 >> > (
		P,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radius_rad,
		tilesKey_touched)
	CHECK_CUDA(, debug)
	// cudaFree(tilesKey_touched);


	int bit = getHigherMsb(tile_grid.x * tile_grid.y * tile_grid.z);
	// std::cout << "Sorting " << num_rendered << " rendered Gaussians..." << std::endl;
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * tile_grid.z * sizeof(uint2)), debug);
	// std::cout << "Identifying tile ranges..." << std::endl;
	// Identify start and end of per-tile workloads in sorted list
	//ranges[0] = (0, 3)  means tile 0 has 3 gaussians from index 0 to 2
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// std::cout << "Rendering..." << std::endl;
	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::renderOmni(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height, N_order, P,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth,
		hp_color,
		hp_depth,
		skip_2d_output), debug)

	// std::cout << "the size of geometry buffer: " << P << std::endl;
	// std::cout << "the size of binning buffer: " << num_rendered << std::endl;
	// std::cout << "the size of image buffer: " << N_pix << std::endl;

	// query_hp_from_equi<<<(width * height + 255) / 256, 256>>>(width, height, N_order, out_color, depth, original_hp, hp_depth);
	
	
	return num_rendered;
}

int CudaRasterizer::FisheyeRasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> healpixBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const int N_side,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	float* hp_color,
	float* hp_depth,
	float* original_image,
	float* original_hp,
	int* radii,
	float* radius_rad,
	bool debug,
	bool skip_2d_output,
	bool use_quadtree_query)
{
	// Fisheye camera mode reuses OmniRasterizer for HEALPix rendering
	// The difference is in Python preprocessing:
	// - Fisheye images are converted to HEALPix GT with a valid pixel mask
	// - Only pixels within the fisheye FOV contribute to the loss
	// The CUDA rendering is identical to omnidirectional mode
	return CudaRasterizer::OmniRasterizer::forward(
		geometryBuffer,
		binningBuffer,
		imageBuffer,
		healpixBuffer,
		P, D, M,
		background,
		width, height,
		means3D,
		shs,
		colors_precomp,
		opacities,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		tan_fovx, tan_fovy,
		N_side,
		prefiltered,
		out_color,
		depth,
		antialiasing,
		hp_color,
		hp_depth,
		original_image,
		original_hp,
		radii,
		radius_rad,
		debug,
		skip_2d_output,
		use_quadtree_query);
}


void CudaRasterizer::OmniRasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const int* radii,
	const float* radius_rad,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	const float* dL_dhppix,
	const float* dL_dhpinvdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dlon_dt,
	float* dlat_dt,
	const int N_side,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, N_side * N_side * 12);

	// const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const dim3 tile_grid((N_side + BLOCK_X_3D - 1) / BLOCK_X_3D, (N_side + BLOCK_Y_3D - 1) / BLOCK_Y_3D, 12);
	const dim3 block(BLOCK_X_3D, BLOCK_Y_3D, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	int N_order = round(log2(N_side));
	// std::cout << "N_order: " << N_order << std::endl;
	CHECK_CUDA(BACKWARD::renderOmni(
		P,
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, N_order,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		dL_dhppix,
		dL_dhpinvdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);
	

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocessOmni(P, D, M,
		(float3*)means3D,
		radii,
		radius_rad,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		(float3*)dlon_dt,
		(float3*)dlat_dt,
		antialiasing), debug);

	
	
}
