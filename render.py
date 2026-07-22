# File: render_dataset.py

# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr

import torch
from scene import Scene
import os
import math
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import resolve_fisheye_params
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, fraction, healpix_scale_render, output_suffix="", fisheye_params=None, N_side=None, perspective_fov=None, perspective_width=None, perspective_height=None, equi_width=None, equi_height=None):
    folder_name = f"{name}_{output_suffix}" if output_suffix else name
    render_path = os.path.join(model_path, folder_name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, folder_name, "ours_{}".format(iteration), "gt")
    
    # For fisheye mode, also save equirectangular renders
    if fisheye_params is not None:
        equi_path = os.path.join(model_path, folder_name, "ours_{}".format(iteration), "equirectangular")
        makedirs(equi_path, exist_ok=True)
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Use the fraction argument to determine the number of views to process
    num_views = int(len(views) * fraction)
    
    progress_bar = tqdm(views[:num_views], desc="Rendering progress")
    for idx, view in enumerate(progress_bar):
        gt_image = view.original_image.cuda()

        render_result = render(view, gaussians, pipeline, gt_image, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, healpix_scale=healpix_scale_render, N_side=N_side)

        if fisheye_params is not None:
            # Fisheye mode: project HEALPix render directly to fisheye
            from scene.cameras import healpix_to_fisheye
            
            hp_render = render_result["hp_color"]
            
            # Save equirectangular render (for reference)
            eq_render = torch.clamp(render_result["render"], 0.0, 1.0)
            torchvision.utils.save_image(eq_render, os.path.join(equi_path, '{0:05d}'.format(idx) + ".png"))
            
            # Project HEALPix directly to fisheye
            rendering = healpix_to_fisheye(
                hp_render,
                fisheye_params['fx'], fisheye_params['fy'],
                fisheye_params['cx'], fisheye_params['cy'],
                fisheye_params['w_x'], fisheye_params['w_y'],
                fisheye_params['W'], fisheye_params['H']
            )
            # Convert GT from equirectangular to fisheye
            from scene.cameras import equirectangular_to_fisheye
            gt_equi = view.original_image[0:3, :, :].cuda()
            gt = equirectangular_to_fisheye(
                gt_equi,
                fisheye_params['fx'], fisheye_params['fy'],
                fisheye_params['cx'], fisheye_params['cy'],
                fisheye_params['w_x'], fisheye_params['w_y'],
                fisheye_params['W'], fisheye_params['H']
            )
        elif perspective_fov is not None:
            # Perspective mode with custom FOV: project HEALPix to perspective
            from scene.cameras import healpix_to_perspective
            
            hp_render = render_result["hp_color"]
            
            # Project to perspective using custom FOV and resolution
            fov_rad = math.radians(perspective_fov)
            # Use custom resolution if provided, otherwise use square output based on smaller dimension
            if perspective_width is not None and perspective_height is not None:
                W, H = perspective_width, perspective_height
            else:
                # Default to 512x512 for perspective output
                W, H = 512, 512
            rendering = healpix_to_perspective(
                hp_render,
                fov_rad, fov_rad,  # FoVx, FoVy (square FOV)
                W, H
            )
            gt = view.original_image[0:3, :, :]
        elif pipeline.camera_model == 0:
            # Perspective mode: project HEALPix back to original perspective view
            # This ensures output matches vanilla 3DGS perspective rendering
            from scene.cameras import healpix_to_perspective
            
            hp_render = render_result["hp_color"]
            
            # Use the view's original FOV and resolution
            rendering = healpix_to_perspective(
                hp_render,
                view.FoVx, view.FoVy,
                view.image_width, view.image_height
            )
            gt = view.original_image[0:3, :, :]
        else:
            # Omnidirectional mode (camera_model==2): output equirectangular
            if equi_width is not None and equi_height is not None:
                # Custom equirectangular resolution: project HEALPix to equirectangular
                from scene.cameras import healpix_to_equirectangular
                hp_render = render_result["hp_color"]
                rendering = healpix_to_equirectangular(hp_render, equi_width, equi_height)
            else:
                # Default: use rasterizer's equirectangular output (same resolution as input)
                rendering = render_result["render"]
            gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torch.cuda.empty_cache()


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, separate_sh: bool, train_fraction: float, test_fraction: float, healpix_scale: int, output_suffix: str = "", perspective_fov: float = None, perspective_width: int = None, perspective_height: int = None, fisheye_width: int = None, fisheye_height: int = None, equi_width: int = None, equi_height: int = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Compute N_side based on image dimensions (same logic as train.py)
        first_cam = scene.getTrainCameras()[0] if scene.getTrainCameras() else scene.getTestCameras()[0]
        W, H = first_cam.image_width, first_cam.image_height
        
        # Calculate N_side_original based on camera model (must match train.py logic)
        if pipeline.camera_model == 0:
            # Perspective camera: calculate equivalent full-sphere resolution based on solid angle
            if perspective_fov is not None:
                # Use custom FOV (same for x and y, square FOV)
                FoVx = FoVy = math.radians(perspective_fov)
                print(f"Using custom perspective FOV: {perspective_fov}° x {perspective_fov}°")
            else:
                FoVx, FoVy = first_cam.FoVx, first_cam.FoVy
                print(f"Perspective FOV (from dataset): {math.degrees(FoVx):.1f}° x {math.degrees(FoVy):.1f}°")
            solid_angle_persp = 4 * np.arcsin(np.sin(FoVx/2) * np.sin(FoVy/2))
            solid_angle_ratio = (4 * np.pi) / solid_angle_persp
            W_eq = int(W * np.sqrt(solid_angle_ratio))
            H_eq = int(H * np.sqrt(solid_angle_ratio))
            print(f"Equivalent full-sphere resolution: {W_eq} x {H_eq} (from perspective {W} x {H})")
            N_side_original = int(2 ** math.ceil(math.log2(round(math.sqrt(W_eq * H_eq / 12.0)))))
        elif pipeline.camera_model == 1:
            # Fisheye camera: calculate equivalent equirectangular resolution based on FoV coverage
            dataset_fisheye_params = resolve_fisheye_params(
                dataset.source_path, W, H,
                pipeline.fisheye_fov_x, pipeline.fisheye_fov_y
            )
            fisheye_fov_x = dataset_fisheye_params['fov_x']
            fisheye_fov_y = dataset_fisheye_params['fov_y']
            fisheye_w_x = dataset_fisheye_params['w_x']
            fisheye_w_y = dataset_fisheye_params['w_y']
            W_eq = int(W * 2.0 / fisheye_w_x)
            H_eq = int(H / fisheye_w_y)
            print(f"Fisheye FoV: {fisheye_fov_x:.1f} degrees x {fisheye_fov_y:.1f} degrees")
            print(f"Equivalent equirectangular resolution: {W_eq} x {H_eq} (from fisheye {W} x {H})")
            N_side_original = int(2 ** math.ceil(math.log2(round(math.sqrt(W_eq * H_eq / 12.0)))))
        else:
            # Omnidirectional (camera_model==2): image is already equirectangular
            N_side_original = int(2 ** math.ceil(math.log2(round(math.sqrt(W * H / 12.0)))))
        
        order_original = int(round(math.log2(N_side_original)))
        order = order_original + healpix_scale
        N_side = int(2 ** order)
        print(f"N_side calculation: W={W}, H={H}, N_side_original={N_side_original}, healpix_scale={healpix_scale}, N_side={N_side}")
        
        # Setup fisheye params if camera_model == 1
        fisheye_params = None
        if pipeline.camera_model == 1:
            first_cam = scene.getTrainCameras()[0] if scene.getTrainCameras() else scene.getTestCameras()[0]
            # Use custom resolution if provided, otherwise use dataset resolution
            if fisheye_width is not None and fisheye_height is not None:
                W, H = fisheye_width, fisheye_height
            else:
                W, H = first_cam.image_width, first_cam.image_height
            
            fisheye_params = resolve_fisheye_params(
                dataset.source_path, W, H,
                pipeline.fisheye_fov_x, pipeline.fisheye_fov_y
            )
            print(f"Fisheye calibration: {fisheye_params['source']}")
            print(f"Fisheye rendering mode: FoV={fisheye_params['fov_x']:.1f} degrees x "
                  f"{fisheye_params['fov_y']:.1f} degrees, "
                  f"fx={fisheye_params['fx']:.2f}, fy={fisheye_params['fy']:.2f}, "
                  f"cx={fisheye_params['cx']:.2f}, cy={fisheye_params['cy']:.2f}, "
                  f"w_x={fisheye_params['w_x']:.6f}, w_y={fisheye_params['w_y']:.6f}, "
                  f"output={W}x{H}")

        if not skip_train:
            train_views = scene.getTrainCameras()
            render_set(dataset.model_path, "train", scene.loaded_iter, train_views, gaussians, pipeline, background, getattr(dataset, 'train_test_exp', False), separate_sh, train_fraction, healpix_scale, output_suffix, fisheye_params, N_side, perspective_fov, perspective_width, perspective_height, equi_width, equi_height)

        if not skip_test:
            test_views = scene.getTestCameras()
            render_set(dataset.model_path, "test", scene.loaded_iter, test_views, gaussians, pipeline, background, getattr(dataset, 'train_test_exp', False), separate_sh, test_fraction, healpix_scale, output_suffix, fisheye_params, N_side, perspective_fov, perspective_width, perspective_height, equi_width, equi_height)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration number to load")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering the training set")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering the testing set")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--train_fraction", default=1.0, type=float, help="Fraction of the training dataset to render (0.0 to 1.0)")
    parser.add_argument("--test_fraction", default=1.0, type=float, help="Fraction of the testing dataset to render (0.0 to 1.0)")
    parser.add_argument("--healpix_scale", type=int, default=0)
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix to append to output folder names (train/test -> train_suffix/test_suffix)")
    parser.add_argument("--perspective_fov", type=float, default=None, help="Custom perspective FOV in degrees (overrides dataset FOV, for visualization only)")
    parser.add_argument("--perspective_width", type=int, default=None, help="Output width for perspective rendering (default: 512)")
    parser.add_argument("--perspective_height", type=int, default=None, help="Output height for perspective rendering (default: 512)")
    parser.add_argument("--fisheye_width", type=int, default=None, help="Output width for fisheye rendering (default: dataset resolution)")
    parser.add_argument("--fisheye_height", type=int, default=None, help="Output height for fisheye rendering (default: dataset resolution)")
    parser.add_argument("--equi_width", type=int, default=None, help="Output width for equirectangular rendering (default: dataset resolution)")
    parser.add_argument("--equi_height", type=int, default=None, help="Output height for equirectangular rendering (default: dataset resolution)")
    args = get_combined_args(parser)


    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        args.train_fraction,
        args.test_fraction,
        args.healpix_scale,
        getattr(args, 'output_suffix', ''),
        getattr(args, 'perspective_fov', None),
        getattr(args, 'perspective_width', None),
        getattr(args, 'perspective_height', None),
        getattr(args, 'fisheye_width', None),
        getattr(args, 'fisheye_height', None),
        getattr(args, 'equi_width', None),
        getattr(args, 'equi_height', None)
    )