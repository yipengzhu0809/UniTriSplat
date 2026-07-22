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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "healpix_util.h"
#include <cstdio>
#include <iostream>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ float3 computeCov2DOmni(const float3& mean, const float* cov3D, const float* viewmatrix, const int width, const int height)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	float trxztrxz = t.x * t.x + t.z * t.z;
	float trxztrxz_inv = 1.0f / (trxztrxz + 0.0000001f);
	float trxz = sqrtf(trxztrxz);
	float trxz_inv = 1.0f / (trxz + 0.0000001f);
	float trtr = trxztrxz + t.y * t.y;
	float trtr_inv = 1.0f / (trtr + 0.0000001f);

	float W_div_2pi = width * 0.5f * M_1_PIf32;
	float H_div_pi = height * M_1_PIf32;

	float dpx_dtx = W_div_2pi * t.z * trxztrxz_inv;
	float dpx_dtz = -W_div_2pi * t.x * trxztrxz_inv;

	float dpy_dtx = -H_div_pi * t.x * t.y * trxz_inv * trtr_inv;
	float dpy_dty = H_div_pi * trxz * trtr_inv;
	float dpy_dtz = -H_div_pi * t.z * t.y * trxz_inv * trtr_inv;

	glm::mat3 J = glm::mat3(
		dpx_dtx, 0.0f, dpx_dtz,
		dpy_dtx, dpy_dty, dpy_dtz,
		0.0f, 0.0f, 0.0f);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ float3 computeCov2DRad(const float3& mean, const float* cov3D, const float* viewmatrix, const int width, const int height)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	float trxztrxz = t.x * t.x + t.z * t.z;
	float trxztrxz_inv = 1.0f / (trxztrxz + 0.0000001f);
	float trxz = sqrtf(trxztrxz);
	float trxz_inv = 1.0f / (trxz + 0.0000001f);
	float trtr = trxztrxz + t.y * t.y;
	float trtr_inv = 1.0f / (trtr + 0.0000001f);

	float dgx_dtx = t.z * trxztrxz_inv;
	float dgx_dty = 0.0f;
	float dgx_dtz = -t.x * trxztrxz_inv;
	float dgy_dtx = -t.x * t.y * trxz_inv * trtr_inv;
	float dgy_dty = trxz * trtr_inv;
	float dgy_dtz = -t.z * t.y * trxz_inv * trtr_inv;

	glm::mat3 J = glm::mat3(
		dgx_dtx, dgx_dty, dgx_dtz,
		dgy_dtx, dgy_dty, dgy_dtz,
		0.0f, 0.0f, 0.0f);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z); // cov.x and cov.z are the diagonal elements, represent σxx and σyy
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template<int C>
__global__ void preprocessCUDAOmni(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const int N_side,
	int* radii,
	float* radius_rad,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	int3* xyf_image,
	int* nest_indices,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P){
		// radii[idx] = -1;
		// radius_rad[idx] = -1.0f;
		// tiles_touched[idx] = 0;
		return;
	}
	
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	radius_rad[idx] = 0.0f;
	tiles_touched[idx] = 0;
	points_xy_image[idx] = {0.0f, 0.0f};
	xyf_image[idx] = {0, 0, 0};
	nest_indices[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view;
	float4 p_view_hom;
	if (omni_culling(p_orig, viewmatrix, p_view, p_view_hom))
		return;

	// Transform point by projecting
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float2 p_proj = p3d2lonlat(p_view);
	// float2 p_proj_screen = point3ToLonlatScreen(p_view_hom); //Keep this for possible future use

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov_rad = computeCov2DRad(p_orig, cov3D, viewmatrix, W, H);
	/**
	 * ------------------------------- For arc cov calculation  ------------------------------------------
	*/
	float lat_scale = cosf(p_proj.y); 
	lat_scale = fmaxf(lat_scale, 1e-6f);

	float3 cov_arc = {
		cov_rad.x * lat_scale * lat_scale,
		cov_rad.y * lat_scale,
		cov_rad.z
	}; // cov in arc length space, in the latitudinal direction need to scale by cos(lat)
	float3 clamp_cov_arc = clampCov22(cov_arc, CLAMP_NUM);
	float a = clamp_cov_arc.x, b = clamp_cov_arc.y, c = clamp_cov_arc.z;
	float det = a * c - b * b;
	det = fmaxf(det,CLAMP_NUM);
	float inv_det = 1.0f / det;
	float3 conic_rad = { c * inv_det, -b * inv_det, a * inv_det };

	float tr   = a + c;
    float diff = a - c;
    float disc = sqrtf(fmaxf(0.0f, diff*diff + 4.0f*b*b));
    float lam_max = 0.5f * (tr + disc);
    lam_max = fmaxf(lam_max, CLAMP_NUM);
    float my_radius_rad  = 3.0f * sqrtf(lam_max);

	float2 lonlat_pt_image = { p_proj.x, p_proj.y }; //the lonlat coordinates
	int ix, iy, face;
	int nest_index;
	healpix_util::lonlat2_nest_xyf(p_proj.x, p_proj.y, ix, iy, face, nest_index, N_side);

	// uint2 rect_min, rect_max;
	// dim3 grid1((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1); //keep for future use 
	// getRect(point_image, my_radius, rect_min, rect_max, grid1);
	// if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
	// 	return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view_hom.w; //the depth should be the Radius in 3D space for omni
	radius_rad[idx] = my_radius_rad;
	// lonlat_image[idx] = lonlat_pt_image;
	// radii[idx] = my_radius; //keep for possible future use
	// printf("Radius of Gaussian %d: %f\n", idx, radii[idx]);
	points_xy_image[idx] = lonlat_pt_image;
	if (points_xy_image[idx].x < -M_PI || points_xy_image[idx].x > M_PI || points_xy_image[idx].y < -M_PI/2 || points_xy_image[idx].y > M_PI/2){
		printf("Error: points_xy_image out of bounds for Gaussian %d: (%f, %f)\n", idx, points_xy_image[idx].x, points_xy_image[idx].y);
	}
	xyf_image[idx] = {ix, iy, face};
	nest_indices[idx] = nest_index;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];
	// conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling }; //keep for possible future use
	conic_opacity[idx] = { 
		conic_rad.x, 
		conic_rad.y, 
		conic_rad.z, 
		opacity
	};
	// tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); //keep for possible future use
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) //this is the maximum number of threads per block, used to optimize register allocation
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X_3D * BLOCK_Y_3D)
renderCUDAOmni(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int order, int P,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth,
	float* __restrict__ hp_color,
	float* __restrict__ hp_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint pix_face = block.group_index().z;
	uint2 pix_min = { block.group_index().x * BLOCK_X_3D, block.group_index().y * BLOCK_Y_3D };
	uint2 pix_max = { pix_min.x + BLOCK_X_3D, pix_min.y + BLOCK_Y_3D};
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint3 pix_id_xyf = { pix.x, pix.y, pix_face };
	uint32_t pix_id;
	healpix_util::xyf2nestCUDA(pix_id_xyf.x, pix_id_xyf.y, pix_id_xyf.z, order, pix_id);
	double pix_z, pix_phi;
	healpix_util::Healpix_Nested_CUDA hp;
	hp.set(order);
	healpix_util::pix2zphiCUDA(int(pix_id),pix_z, pix_phi, hp);
	if (pix_phi < 0){
		pix_phi = 0.00001f;
	}
	if (pix_phi > 2.0 * M_PIf){
		pix_phi = 2.0f * M_PIf - 0.00001f;
	}
	double pix_theta = acos(pix_z);
	float pix_lat = M_PIf / 2.0f - pix_theta;
	float pix_lon = pix_phi - M_PIf;


	bool inside = true;
	bool done = !inside;
	int N_side = 1 << order;
	uint32_t horizontal_blocks = (N_side + BLOCK_X_3D - 1) / BLOCK_X_3D;
	int block_order = round(log2(horizontal_blocks));
	uint32_t block_index;
	healpix_util::xyf2nestCUDA(block.group_index().x, block.group_index().y, block.group_index().z, block_order, block_index);
	uint2 range = ranges[block_index];
	const int rounds = ((range.y - range.x + BLOCK_SIZE_3D - 1) / BLOCK_SIZE_3D);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE_3D];
	__shared__ float2 collected_xy[BLOCK_SIZE_3D];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE_3D];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE_3D)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE_3D)
			break;
		int progress = i * BLOCK_SIZE_3D + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync(); //all the threads in the block synchronize here

		for (int j = 0; !done && j < min(BLOCK_SIZE_3D, toDo); j++){
			contributor++;

			int cid = collected_id[j];
			if (cid < 0 || cid >= P) {
                if (block.thread_rank() == 0 && j == 0) {
                    printf("ERROR: invalid collected_id[%d] = %d (P=%d) in block (%d,%d,%d)\n",
                           j, cid, P, block.group_index().x, block.group_index().y, block.group_index().z);
                }
				// printf("ERROR: invalid collected_id[%d] = %d (P=%d) in block (%d,%d,%d)\n",
				// 	   j, cid, P, block.group_index().x, block.group_index().y, block.group_index().z);
				// CUDA_error("Invalid collected_id");
                continue;
            }

			float2 lonlat = collected_xy[j]; //This is now actually the lonlat coordinates
			float2 pix_lonlat = {pix_lon, pix_lat};
			float great_circle_rad, bearing;
/**
	 * ------------------------------- For arc cov calculation  ------------------------------------------
	*/
			//set the great circle distance between two lonlat points
			greatCircleDistance(lonlat, pix_lonlat, great_circle_rad);
			//calcualte the bearing of the great circle from lonlat to pix_lonlat
			computeBearing(pix_lonlat, lonlat, bearing);
			// computeBearing(lonlat, pix_lonlat, bearing);
			// Calculate the vector on the local sectional plane
			float dy = great_circle_rad * cosf(bearing);  // North-south component
			float dx = great_circle_rad * sinf(bearing);  // East-west component
/**
	 * ------------------------------- For rad cov calculation  ------------------------------------------
	*/
			// float dx = shortest_angular_difference(pix_lonlat.x, lonlat.x);
			// float dy = lonlat.y - pix_lonlat.y;
/**
	 * ---------------------------------------------------------------------------------------------------------
	*/
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * dx * dx + con_o.z * dy * dy) - con_o.y * dx * dy;
			if (power > 0.0f)
				continue;
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			for (int ch = 0; ch < CHANNELS; ch++){
				if ((size_t)(cid) * CHANNELS + ch < (size_t)P * CHANNELS) {
                    C[ch] += features[cid * CHANNELS + ch] * alpha * T;
                }
				else{
					printf("ERROR: invalid memory access features[%d] in block (%d,%d,%d)\n",
						   cid * CHANNELS + ch, block.group_index().x, block.group_index().y, block.group_index().z);
				}
				// C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}

			if(invdepth){
				// expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;
				if (cid >= 0 && cid < P) expected_invdepth += (1.0f / depths[cid]) * alpha * T;
			}

			T = test_T;
			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// float2 equi;
	// lonlat2Equipixel(pix_lon, pix_lat, W, H, equi);
	// uint32_t pix_equi_id = (uint32_t)equi.y * W + (uint32_t)equi.x;
	// if (pix_equi_id >= W * H){
	// 	inside = false;
	// }
	if (pix_id >= N_side * N_side * 12){
		inside = false;
	}
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++){
			// out_color[ch * H * W + pix_equi_id] = C[ch] + T * bg_color[ch];
			hp_color[ch * N_side * N_side * 12 + pix_id] = C[ch] + T * bg_color[ch];
		}

		if (invdepth){
			// invdepth[pix_equi_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
			hp_depth[pix_id] = expected_invdepth;
		}
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}

void FORWARD::renderOmni(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int order, int P,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth,
	float* hp_color,
	float* hp_depth,
	bool skip_2d_output)
{
	renderCUDAOmni<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H, order, P,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth,
		hp_color,
		hp_depth);

	// Skip query_hp_from_equi during training when 2D output is not needed
	if (!skip_2d_output) {
		query_hp_from_equi<<<(W * H + 255) / 256, 256>>>(W, H, order, out_color, depth, hp_color, hp_depth);
	}
	
}


void FORWARD::preprocessOmni(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const int N_side,
	int* radii,
	float* radius_rad,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	int3* xyf_image,
	int* nest_indices,
	bool prefiltered,
	dim3 block,
	bool antialiasing)
{

	preprocessCUDAOmni<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		N_side,
		radii,
		radius_rad,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		xyf_image,
		nest_indices,
		prefiltered,
		antialiasing
		);


}
