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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug,
	const int camera_model,
	const torch::Tensor& original_image,
	const int healpix_scale,
	const int N_side,  // N_side now passed from Python
	const bool skip_2d_output,
	const bool use_quadtree_query)  // Use quadtree method (slower but more accurate)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
//   printf("Rasterizing %d Gaussians to image of size %dx%d\n", P, W, H);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  torch::Tensor healpixBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> healpixFunc = resizeFunctional(healpixBuffer);
  

//the tensor used to maintain the healpix map for omnidirectional camera model
  torch::Tensor radius_rad = torch::full({P}, 0.0f, means3D.options().dtype(torch::kFloat32));
//   const int N_side = 256;
  // N_side is now passed from Python, no need to calculate here
//   printf("N_side used in rendering: %d \n", N_side);
  torch::Tensor hp_color = torch::full({NUM_CHANNELS, 12 * N_side * N_side}, 0.0, float_opts);
  torch::Tensor original_hp = torch::full({NUM_CHANNELS, 12 * N_side * N_side}, 0.0, float_opts);
  float* hp_invdepthptr = nullptr;
  torch::Tensor hp_invdepth = torch::full({0, 12 * N_side * N_side}, 0.0, float_opts);
  hp_invdepth = torch::full({1, 12 * N_side * N_side}, 0.0, float_opts).contiguous();
  hp_invdepthptr = hp_invdepth.data<float>();
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  if(camera_model == 0) //pinhole - now uses OmniRasterizer for HEALPix rendering
	  {
		rendered = CudaRasterizer::OmniRasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			healpixFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(), 
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(), 
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			N_side,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_invdepthptr,
			antialiasing,
			hp_color.contiguous().data<float>(),
			hp_invdepthptr,
			original_image.contiguous().data<float>(),
			original_hp.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			radius_rad.contiguous().data<float>(),
			debug,
			skip_2d_output,
			use_quadtree_query);
	  }
	  else if(camera_model == 1) //fisheye
	  {
		rendered = CudaRasterizer::FisheyeRasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			healpixFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(), 
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(), 
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			N_side,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_invdepthptr,
			antialiasing,
			hp_color.contiguous().data<float>(),
			hp_invdepthptr,
			original_image.contiguous().data<float>(),
			original_hp.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			radius_rad.contiguous().data<float>(),
			debug,
			skip_2d_output,
			use_quadtree_query);
	  }
	  else if(camera_model == 2) //omnidirectional
	  {
		rendered = CudaRasterizer::OmniRasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			healpixFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(), 
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(), 
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			N_side,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_invdepthptr,
			antialiasing,
			hp_color.contiguous().data<float>(),
			hp_invdepthptr,
			original_image.contiguous().data<float>(),
			original_hp.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			radius_rad.contiguous().data<float>(),
			debug,
			skip_2d_output,
			use_quadtree_query);
	  }


  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth, hp_color, hp_invdepth, radius_rad, original_hp);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& radius_rad,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& dL_dout_hp_color,
	const torch::Tensor& dL_dout_hp_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug,
	const int camera_model,
	const int healpix_scale,
	const int N_side)  // N_side now passed from Python
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
//   const int N_side = 256;
  // N_side is now passed from Python, no need to calculate here
//   std::cout<<"N_side is: " << N_side << std::endl;

//   std::cout<< "the size of the image buffer is: " << imageBuffer.numel() << std::endl;
//   std::cout<< "the size of the binning buffer is: " << binningBuffer.numel() << std::endl;
//   std::cout<< "the size of the geom buffer is: " << geomBuffer.numel() << std::endl;
//   std::cout<<"the Height and Width are: " << H << " , " << W << std::endl;
  

  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options()); // the gaussian level colors
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  torch::Tensor dL_dhpinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  float* dL_douthpinvdepthptr = nullptr;
  // Enable depth gradients if either equirectangular or HEALPix depth gradients are provided
  if(dL_dout_invdepth.size(0) != 0 || dL_dout_hp_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	if(dL_dout_invdepth.size(0) != 0)
		dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
	if(dL_dout_hp_invdepth.size(0) != 0)
		dL_douthpinvdepthptr = dL_dout_hp_invdepth.data<float>();
  }

  if(P != 0)
  {  
	if(camera_model == 0) //pinhole - now uses OmniRasterizer for HEALPix backward
	{
		torch::Tensor dlon_dt = torch::zeros({P, 3}, means3D.options());
		torch::Tensor dlat_dt = torch::zeros({P, 3}, means3D.options());
		CudaRasterizer::OmniRasterizer::backward(P, degree, M, R,
			background.contiguous().data<float>(),
			W, H, 
			means3D.contiguous().data<float>(),
			sh.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			opacities.contiguous().data<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			radius_rad.contiguous().data<float>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data<float>(),
			dL_dout_invdepthptr, // pinhole level invdepth gradients
			dL_dout_hp_color.contiguous().data<float>(),
			dL_douthpinvdepthptr, // healpix level invdepth gradients
			dL_dmeans2D.contiguous().data<float>(),
			dL_dconic.contiguous().data<float>(),  
			dL_dopacity.contiguous().data<float>(),
			dL_dcolors.contiguous().data<float>(),
			dL_dinvdepthsptr, // Gaussian level invdepth gradients
			dL_dmeans3D.contiguous().data<float>(),
			dL_dcov3D.contiguous().data<float>(),
			dL_dsh.contiguous().data<float>(),
			dL_dscales.contiguous().data<float>(),
			dL_drotations.contiguous().data<float>(),
			dlon_dt.contiguous().data<float>(),
			dlat_dt.contiguous().data<float>(),
			N_side,
			antialiasing,
			debug);
	}
	else if(camera_model == 2) //omnidirectional
	{
		torch::Tensor dlon_dt = torch::zeros({P, 3}, means3D.options());
		torch::Tensor dlat_dt = torch::zeros({P, 3}, means3D.options());
		CudaRasterizer::OmniRasterizer::backward(P, degree, M, R,
			background.contiguous().data<float>(),
			W, H, 
			means3D.contiguous().data<float>(),
			sh.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			opacities.contiguous().data<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			radius_rad.contiguous().data<float>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data<float>(),
			dL_dout_invdepthptr, // pinhole level invdepth gradients
			dL_dout_hp_color.contiguous().data<float>(),
			dL_douthpinvdepthptr, // healpix level invdepth gradients
			dL_dmeans2D.contiguous().data<float>(),
			dL_dconic.contiguous().data<float>(),  
			dL_dopacity.contiguous().data<float>(),
			dL_dcolors.contiguous().data<float>(),
			dL_dinvdepthsptr, // Gaussian level invdepth gradients
			dL_dmeans3D.contiguous().data<float>(),
			dL_dcov3D.contiguous().data<float>(),
			dL_dsh.contiguous().data<float>(),
			dL_dscales.contiguous().data<float>(),
			dL_drotations.contiguous().data<float>(),
			dlon_dt.contiguous().data<float>(),
			dlat_dt.contiguous().data<float>(),
			N_side,
			antialiasing,
			debug);
	}
	else if(camera_model == 1) //fisheye - reuses OmniRasterizer backward
	{
		torch::Tensor dlon_dt = torch::zeros({P, 3}, means3D.options());
		torch::Tensor dlat_dt = torch::zeros({P, 3}, means3D.options());
		CudaRasterizer::OmniRasterizer::backward(P, degree, M, R,
			background.contiguous().data<float>(),
			W, H, 
			means3D.contiguous().data<float>(),
			sh.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			opacities.contiguous().data<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			radius_rad.contiguous().data<float>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data<float>(),
			dL_dout_invdepthptr, // pinhole level invdepth gradients
			dL_dout_hp_color.contiguous().data<float>(),
			dL_douthpinvdepthptr, // healpix level invdepth gradients
			dL_dmeans2D.contiguous().data<float>(),
			dL_dconic.contiguous().data<float>(),  
			dL_dopacity.contiguous().data<float>(),
			dL_dcolors.contiguous().data<float>(),
			dL_dinvdepthsptr, // Gaussian level invdepth gradients
			dL_dmeans3D.contiguous().data<float>(),
			dL_dcov3D.contiguous().data<float>(),
			dL_dsh.contiguous().data<float>(),
			dL_dscales.contiguous().data<float>(),
			dL_drotations.contiguous().data<float>(),
			dlon_dt.contiguous().data<float>(),
			dlat_dt.contiguous().data<float>(),
			N_side,
			antialiasing,
			debug);
	}
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
