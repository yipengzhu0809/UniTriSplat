#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import resolve_fisheye_params
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from healpix_ssim import healpix_ssim
    HEALPIX_SSIM_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] healpix_ssim import failed: {e}")
    HEALPIX_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, healpix_scale, healpix_scale_test):
    if TENSORBOARD_FOUND:
        print("Tensorboard is available: logging progress")
    else:
        print("Tensorboard not available: not logging progress")

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, pipe)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Load healpix mask if present (pixels with mask=0 are excluded from training)
    healpix_mask = None
    interior_mask = None  # Eroded mask for SSIM (excludes edge pixels)
    mask_path = os.path.join(dataset.source_path, "mask_healpix.npy") if hasattr(dataset, "source_path") else "mask_healpix.npy"
    if os.path.exists(mask_path):
        try:
            mask_np = np.load(mask_path)
            healpix_mask = torch.from_numpy(mask_np.astype(np.float32)).cuda()
            print(f"Loaded healpix mask from {mask_path}, shape={healpix_mask.shape}")
        except Exception as e:
            print(f"Failed to load healpix mask '{mask_path}': {e}")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    first_cam = scene.getTrainCameras()[0]
    W, H = first_cam.image_width, first_cam.image_height
    
    # Calculate N_side based on equivalent equirectangular resolution
    perspective_params = None  # Will be set if camera_model == 0
    fisheye_params = None  # Will be set if camera_model == 1
    if pipe.camera_model == 1:
        fisheye_params = resolve_fisheye_params(
            dataset.source_path, W, H,
            pipe.fisheye_fov_x, pipe.fisheye_fov_y
        )

    if pipe.camera_model == 0:
        # For perspective camera, calculate equivalent full panorama resolution
        # Perspective covers FoVx x FoVy radians
        # Use solid angle ratio to estimate equivalent HEALPix resolution
        FoVx = first_cam.FoVx
        FoVy = first_cam.FoVy
        
        # Solid angle of perspective: 4 * arcsin(sin(FoVx/2) * sin(FoVy/2))
        # Solid angle of full sphere: 4π
        solid_angle_persp = 4 * np.arcsin(np.sin(FoVx/2) * np.sin(FoVy/2))
        solid_angle_ratio = (4 * np.pi) / solid_angle_persp
        W_eq = int(W * np.sqrt(solid_angle_ratio))
        H_eq = int(H * np.sqrt(solid_angle_ratio))
        
        print(f"Perspective FOV: {math.degrees(FoVx):.1f}° x {math.degrees(FoVy):.1f}°")
        print(f"Equivalent full-sphere resolution: {W_eq} x {H_eq} (from perspective {W} x {H})")
        
        N_side_original = int(2 ** math.ceil(math.log2(round(math.sqrt(W_eq * H_eq / 12.0)))))
        
        # Store perspective params for later use
        perspective_params = {
            'FoVx': FoVx, 'FoVy': FoVy,
            'W': W, 'H': H
        }
    elif pipe.camera_model == 1:
        # For fisheye camera, calculate equivalent full panorama resolution
        # Fisheye covers approximately (pi * w_x) x (pi * w_y) radians
        # Equirectangular covers 2*pi x pi radians
        # Scale fisheye resolution to equivalent equirectangular resolution
        fisheye_fov_x = fisheye_params['fov_x']
        fisheye_fov_y = fisheye_params['fov_y']
        fisheye_w_x = fisheye_params['w_x']
        fisheye_w_y = fisheye_params['w_y']
        
        # Preserve the resolution rule used by the original experiments.
        W_eq = int(W * 2.0 / fisheye_w_x)
        H_eq = int(H / fisheye_w_y)
        
        print(f"Fisheye FoV: {fisheye_fov_x:.1f} degrees x {fisheye_fov_y:.1f} degrees")
        print(f"Equivalent equirectangular resolution: {W_eq} x {H_eq} (from fisheye {W} x {H})")
        
        N_side_original = int(2 ** math.ceil(math.log2(round(math.sqrt(W_eq * H_eq / 12.0)))))
    else:
        N_side_original = int(2 ** math.ceil(math.log2(round(math.sqrt(W * H / 12.0)))))
    
    order_original = int(round(math.log2(N_side_original)))
    order = order_original + healpix_scale
    N_side = int(2 ** order)

    order_test = order_original + healpix_scale_test if healpix_scale_test is not None else order
    N_side_test = int(2 ** order_test)

    eval_train_indices = [idx % len(scene.getTrainCameras()) for idx in range(5, 30, 5)]
    eval_train_cameras = [scene.getTrainCameras()[idx] for idx in eval_train_indices]

    precomputed_hp_gt = {} 
    precomputed_hp_gt_test = {} 
    precomputed_hp_depth = {}  # Precomputed HEALPix depth GT
    
    # Handle HEALPix GT precomputation based on camera model
    if pipe.camera_model == 0:
        # Perspective mode: convert perspective images to HEALPix and generate mask
        from scene.cameras import perspective_to_healpix_gt, generate_perspective_healpix_mask, perspective_depth_to_healpix
        
        FoVx = first_cam.FoVx
        FoVy = first_cam.FoVy
        
        print(f"Perspective camera parameters: FoVx={math.degrees(FoVx):.1f}°, FoVy={math.degrees(FoVy):.1f}°")
        print(f"Using HEALPix rendering for perspective camera (N_side={N_side})")
        
        # Generate HEALPix mask for perspective (pixels visible in perspective image)
        # Note: Each camera may have different FoV, but we assume same intrinsics for now
        if healpix_mask is None:
            print(f"Generating HEALPix mask for perspective camera (N_side={N_side})...")
            healpix_mask = generate_perspective_healpix_mask(
                FoVx, FoVy, W, H, N_side
            ).float().cuda()
            print(f"Perspective HEALPix mask generated: {healpix_mask.sum().item()}/{healpix_mask.numel()} valid pixels")
            
        
        print(f"Precomputing HEALPix GT for {len(scene.getTrainCameras())} training cameras (N_side={N_side})...")
        for cam in tqdm(scene.getTrainCameras(), desc="Precomputing HEALPix GT (perspective train)"):
            hp_gt, _ = perspective_to_healpix_gt(
                cam.original_image, cam.FoVx, cam.FoVy, N_side
            )
            precomputed_hp_gt[cam.image_name] = hp_gt
            # Precompute HEALPix depth if available
            if cam.invdepthmap is not None and cam.depth_reliable:
                hp_depth, _ = perspective_depth_to_healpix(cam.invdepthmap, cam.FoVx, cam.FoVy, N_side)
                precomputed_hp_depth[cam.image_name] = hp_depth
        print(f"HEALPix GT precomputation done. Shape per camera: {precomputed_hp_gt[first_cam.image_name].shape}")
        if precomputed_hp_depth:
            print(f"HEALPix Depth precomputation done for {len(precomputed_hp_depth)} cameras.")
        
        if healpix_scale_test is not None:
            # Generate test mask with N_side_test if different from N_side
            if N_side_test != N_side:
                print(f"Generating HEALPix mask for testing (N_side_test={N_side_test})...")
                healpix_mask_test = generate_perspective_healpix_mask(
                    FoVx, FoVy, W, H, N_side_test
                ).float().cuda()
                print(f"Test HEALPix mask generated: {healpix_mask_test.sum().item()}/{healpix_mask_test.numel()} valid pixels")
            else:
                healpix_mask_test = healpix_mask
            
            all_test_cameras = list(scene.getTestCameras()) + eval_train_cameras
            print(f"Precomputing HEALPix GT for {len(all_test_cameras)} test/eval cameras (N_side_test={N_side_test})...")
            for cam in tqdm(all_test_cameras, desc="Precomputing HEALPix GT (perspective test)"):
                if cam.image_name not in precomputed_hp_gt_test:
                    hp_gt, _ = perspective_to_healpix_gt(
                        cam.original_image, cam.FoVx, cam.FoVy, N_side_test
                    )
                    precomputed_hp_gt_test[cam.image_name] = hp_gt
            print(f"HEALPix GT precomputation (test) done.")
        else:
            healpix_mask_test = healpix_mask
    
    elif pipe.camera_model == 1:
        # Fisheye mode: convert fisheye images to HEALPix and generate mask
        from scene.cameras import fisheye_to_healpix_gt, generate_fisheye_healpix_mask
        
        fisheye_fx = fisheye_params['fx']
        fisheye_fy = fisheye_params['fy']
        fisheye_cx = fisheye_params['cx']
        fisheye_cy = fisheye_params['cy']
        fisheye_w_x = fisheye_params['w_x']
        fisheye_w_y = fisheye_params['w_y']
        
        print(f"Fisheye calibration: {fisheye_params['source']}")
        print(f"Fisheye FoV: {fisheye_params['fov_x']:.1f} degrees x "
              f"{fisheye_params['fov_y']:.1f} degrees")
        print(f"Derived fisheye parameters: fx={fisheye_fx:.1f}, fy={fisheye_fy:.1f}, "
              f"cx={fisheye_cx:.1f}, cy={fisheye_cy:.1f}, "
              f"w_x={fisheye_w_x:.6f}, w_y={fisheye_w_y:.6f}")
        
        # Generate HEALPix mask for fisheye (pixels visible in fisheye image)
        # This mask is shared by all cameras since they have the same intrinsics
        if healpix_mask is None:
            print(f"Generating HEALPix mask for fisheye camera (N_side={N_side})...")
            healpix_mask = generate_fisheye_healpix_mask(
                fisheye_fx, fisheye_fy, fisheye_cx, fisheye_cy, 
                fisheye_w_x, fisheye_w_y, W, H, N_side
            ).float().cuda()
            print(f"Fisheye HEALPix mask generated: {healpix_mask.sum().item()}/{healpix_mask.numel()} valid pixels")
            
        
        print(f"Precomputing HEALPix GT for {len(scene.getTrainCameras())} training cameras (N_side={N_side})...")
        for cam in tqdm(scene.getTrainCameras(), desc="Precomputing HEALPix GT (fisheye train)"):
            hp_gt, _ = fisheye_to_healpix_gt(
                cam.original_image, fisheye_fx, fisheye_fy, 
                fisheye_cx, fisheye_cy, fisheye_w_x, fisheye_w_y, N_side
            )
            precomputed_hp_gt[cam.image_name] = hp_gt
        print(f"HEALPix GT precomputation done. Shape per camera: {precomputed_hp_gt[first_cam.image_name].shape}")
        
        if healpix_scale_test is not None:
            # Generate test mask with test N_side
            print(f"Generating HEALPix mask for fisheye camera (N_side_test={N_side_test})...")
            healpix_mask_test = generate_fisheye_healpix_mask(
                fisheye_fx, fisheye_fy, fisheye_cx, fisheye_cy, 
                fisheye_w_x, fisheye_w_y, W, H, N_side_test
            ).float().cuda()
            print(f"Fisheye HEALPix test mask generated: {healpix_mask_test.sum().item()}/{healpix_mask_test.numel()} valid pixels")
            
            all_test_cameras = list(scene.getTestCameras()) + eval_train_cameras
            print(f"Precomputing HEALPix GT for {len(all_test_cameras)} test/eval cameras (N_side_test={N_side_test})...")
            for cam in tqdm(all_test_cameras, desc="Precomputing HEALPix GT (fisheye test)"):
                if cam.image_name not in precomputed_hp_gt_test:
                    hp_gt, _ = fisheye_to_healpix_gt(
                        cam.original_image, fisheye_fx, fisheye_fy,
                        fisheye_cx, fisheye_cy, fisheye_w_x, fisheye_w_y, N_side_test
                    )
                    precomputed_hp_gt_test[cam.image_name] = hp_gt
            print(f"HEALPix GT precomputation (test) done.")
        else:
            healpix_mask_test = healpix_mask
    
    elif pipe.camera_model == 2:
        # Omnidirectional (equirectangular) mode: original logic
        from scene.cameras import equirectangular_to_healpix_gt, equirectangular_depth_to_healpix
        
        print(f"Precomputing HEALPix GT for {len(scene.getTrainCameras())} training cameras (N_side={N_side})...")
        for cam in tqdm(scene.getTrainCameras(), desc="Precomputing HEALPix GT (train)"):
            precomputed_hp_gt[cam.image_name] = equirectangular_to_healpix_gt(cam.original_image, N_side)
            # Precompute HEALPix depth if available
            if cam.invdepthmap is not None and cam.depth_reliable:
                precomputed_hp_depth[cam.image_name] = equirectangular_depth_to_healpix(cam.invdepthmap, N_side)
        print(f"HEALPix GT precomputation done. Shape per camera: {precomputed_hp_gt[first_cam.image_name].shape}")
        if precomputed_hp_depth:
            print(f"HEALPix Depth precomputation done for {len(precomputed_hp_depth)} cameras.")
        
        if healpix_scale_test is not None:
            all_test_cameras = list(scene.getTestCameras()) + eval_train_cameras
            print(f"Precomputing HEALPix GT for {len(all_test_cameras)} test/eval cameras (N_side_test={N_side_test})...")
            for cam in tqdm(all_test_cameras, desc="Precomputing HEALPix GT (test)"):
                if cam.image_name not in precomputed_hp_gt_test:
                    precomputed_hp_gt_test[cam.image_name] = equirectangular_to_healpix_gt(cam.original_image, N_side_test)
            print(f"HEALPix GT precomputation (test) done.")
            healpix_mask_test = healpix_mask  # For omni, full sphere is valid, so same mask
        else:
            healpix_mask_test = healpix_mask

    # Generate interior_mask if healpix_mask exists but interior_mask doesn't
    # This handles all camera models (perspective, fisheye, omnidirectional)
    if healpix_mask is not None and interior_mask is None:
        import healpy as hp
        print(f"Generating interior mask by eroding healpix_mask...")
        mask_np = healpix_mask.cpu().numpy()
        mask_ring = hp.reorder(mask_np, n2r=True)
        
        # Efficient erosion using vectorized neighbor lookup
        K_erode = 5  # Same as SSIM kernel radius
        interior_ring = mask_ring.copy()
        npix = len(mask_ring)
        for k in range(K_erode):
            all_neighbors = hp.get_all_neighbours(N_side, np.arange(npix), nest=False)
            valid_neighbor_mask = all_neighbors >= 0
            all_neighbors_clamped = np.clip(all_neighbors, 0, npix - 1)
            neighbor_vals = interior_ring[all_neighbors_clamped]
            neighbor_vals = np.where(valid_neighbor_mask, neighbor_vals, 1.0)
            interior_ring = interior_ring * np.min(neighbor_vals, axis=0)
        
        interior_nested = hp.reorder(interior_ring, r2n=True)
        interior_mask = torch.from_numpy(interior_nested.astype(np.float32)).cuda()
        print(f"Interior mask generated: {interior_mask.sum().item():.0f}/{interior_mask.numel()} interior pixels "
              f"({100*interior_mask.sum().item()/healpix_mask.sum().item():.1f}% of valid)")
        

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        hp_color = None

        # All camera models now use HEALPix rendering
        # Pass empty tensor since C++ expects torch.Tensor type (not used for HEALPix rendering)
        gt_image = torch.empty(0, device="cuda")

        # Skip 2D output generation during training to save time (all modes use HEALPix)
        skip_2d = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, gt_image, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, healpix_scale=healpix_scale, N_side=N_side, skip_2d_output=skip_2d)
        # All camera models now use HEALPix outputs
        image, viewspace_point_tensor, visibility_filter, radii, hp_color, hp_invdepth, radius_rad, original_hp = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["hp_color"], render_pkg["hp_invdepth"], render_pkg["radius_rad"], render_pkg["original_hp"]
        
        # Use precomputed HEALPix GT for all camera models
        if viewpoint_cam.image_name in precomputed_hp_gt:
            original_hp = precomputed_hp_gt[viewpoint_cam.image_name].cuda()
        
        # Prepare healpix mask for loss computation (DO NOT pre-multiply images!)
        # Pre-multiplying would corrupt SSIM neighborhood statistics
        hp_mask_for_loss = None
        if healpix_mask is not None:
            mask_t = healpix_mask.to(dtype=hp_color.dtype, device=hp_color.device)
            # hp_color shape is (C, N) where C=3 channels, N=num_pixels
            # mask shape is (N,) - expand to (1, N) then broadcast to (C, N)
            if mask_t.dim() == 1 and hp_color.dim() == 2:
                mask_t = mask_t.unsqueeze(0)  # (N,) -> (1, N), will broadcast to (C, N)
            hp_mask_for_loss = mask_t
            # NOTE: Do NOT multiply hp_color and original_hp by mask here!
            # SSIM needs the original pixel values for correct neighborhood statistics
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        # Loss - All camera models now use HEALPix loss
        # Strategy: Edge region uses 100% L1, interior region uses L1+SSIM
        # This avoids SSIM edge artifacts while still using L1 everywhere
        Ll1, ssim_value, loss = None, None, None
        
        if interior_mask is not None and hp_mask_for_loss is not None:
            # Separate edge and interior regions
            interior_mask_t = interior_mask.to(dtype=hp_color.dtype, device=hp_color.device)
            if interior_mask_t.dim() == 1:
                interior_mask_t = interior_mask_t.unsqueeze(0)  # (N,) -> (1, N)
            
            # Edge mask = valid pixels that are NOT interior
            edge_mask_t = hp_mask_for_loss - interior_mask_t  # (1, N)
            edge_mask_t = edge_mask_t.clamp(min=0)  # Handle any floating point issues
            
            # Count pixels
            num_interior = interior_mask_t.sum() * hp_color.shape[0]
            num_edge = edge_mask_t.sum() * hp_color.shape[0]
            num_total_valid = num_interior + num_edge
            
            # L1 loss for edge region (100% weight)
            edge_l1 = torch.abs(hp_color * edge_mask_t - original_hp * edge_mask_t).sum() / num_edge.clamp(min=1.0)
            
            # L1 loss for interior region
            interior_l1 = torch.abs(hp_color * interior_mask_t - original_hp * interior_mask_t).sum() / num_interior.clamp(min=1.0)
            
            # SSIM for interior region only
            if HEALPIX_SSIM_AVAILABLE:
                ssim_interior_mask = interior_mask.to(dtype=hp_color.dtype, device=hp_color.device)
                ssim_value = healpix_ssim(hp_color, original_hp, train=True, mask=ssim_interior_mask)
            else:
                ssim_value = torch.tensor(1.0, device="cuda")  # No SSIM contribution if not available
            
            # Combined loss:
            # Edge: 100% L1
            # Interior: (1-λ)*L1 + λ*(1-SSIM)
            interior_loss = (1.0 - opt.lambda_dssim) * interior_l1 + opt.lambda_dssim * (1.0 - ssim_value)
            edge_loss = edge_l1
            
            # Weighted average by pixel count
            loss = (num_interior * interior_loss + num_edge * edge_loss) / num_total_valid.clamp(min=1.0)
            Ll1 = (num_interior * interior_l1 + num_edge * edge_l1) / num_total_valid.clamp(min=1.0)
        else:
            # No mask or no interior mask: use standard loss
            Ll1 = l1_loss(hp_color, original_hp)
            if HEALPIX_SSIM_AVAILABLE:
                ssim_value = healpix_ssim(hp_color, original_hp, train=True, mask=None)
            else:
                ssim_value = ssim(image, gt_image) if gt_image.numel() > 0 else torch.tensor(0.0, device="cuda")
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        # Depth regularization - All camera models now use HEALPix depth
        Ll1depth_pure = 0.0

        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            # All camera models use HEALPix depth
            hp_invdepth_render = render_pkg["hp_invdepth"]  # (1, N_pix)
            if viewpoint_cam.image_name in precomputed_hp_depth:
                hp_depth_gt = precomputed_hp_depth[viewpoint_cam.image_name].cuda()  # (1, N_pix)
                # Apply healpix mask if present
                if healpix_mask is not None:
                    mask_t = healpix_mask.to(dtype=hp_invdepth_render.dtype, device=hp_invdepth_render.device)
                    if mask_t.dim() == 1:
                        mask_t = mask_t.unsqueeze(0)  # (N,) -> (1, N)
                    # Compute masked L1 loss: sum of |diff| / number of valid pixels
                    depth_diff = torch.abs(hp_invdepth_render - hp_depth_gt) * mask_t
                    num_valid = mask_t.sum().clamp(min=1.0)
                    Ll1depth_pure = depth_diff.sum() / num_valid
                else:
                    # No mask: use all pixels
                    Ll1depth_pure = torch.abs(hp_invdepth_render - hp_depth_gt).mean()
            else:
                Ll1depth_pure = 0.0
            
            if isinstance(Ll1depth_pure, float):
                Ll1depth = 0
            else:
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
                
            if iteration == opt.iterations:
                progress_bar.close()

            # Use healpix_mask_test for evaluation (matches N_side_test)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene, render, (pipe, gt_image, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp, healpix_scale_test, N_side_test), dataset.train_test_exp, healpix_mask_test, precomputed_hp_gt_test, eval_train_cameras, fisheye_params, perspective_params)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
            # All camera models now use HEALPix-based densification with radius_rad
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in radian-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radius_rad[visibility_filter])
                # Use 3D gradient for HEALPix rendering (more accurate than 2D spherical coords)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, use_3d_grad=False)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold in radians (convert 20 pixels to radians)
                    size_threshold = min(20 / viewpoint_cam.image_width * 2 * 3.141592653589793, 20 / viewpoint_cam.image_height * 3.141592653589793) if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radius_rad)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radius_rad > 1e-7
                    gaussians.optimizer.step(visible, radius_rad.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args, pipe):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    config = vars(args).copy()
    config.update(vars(pipe))
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**config)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, healpix_mask=None, precomputed_hp_gt_test=None, eval_train_cameras=None, fisheye_params=None, perspective_params=None):
    """
    fisheye_params: dict with keys 'fx', 'fy', 'cx', 'cy', 'w_x', 'w_y', 'W', 'H' for fisheye projection back
    perspective_params: dict with keys 'FoVx', 'FoVy', 'W', 'H' for perspective projection from HEALPix
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if eval_train_cameras is None:
            eval_train_cameras = [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : eval_train_cameras})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    pipe, _, background, scaling_mod, separate_sh, override_color, use_trained_exp, healpix_scale_test, N_side_test = renderArgs
                    current_gt_image = viewpoint.original_image.cuda()
                    render_result = renderFunc(viewpoint, scene.gaussians, pipe, current_gt_image, background, scaling_mod, separate_sh, override_color, use_trained_exp, healpix_scale_test, N_side_test)
                    if "hp_color" in render_result:
                        image = render_result["hp_color"]
                        if precomputed_hp_gt_test is not None and viewpoint.image_name in precomputed_hp_gt_test:
                            gt_image = precomputed_hp_gt_test[viewpoint.image_name].cuda()
                        else:
                            gt_image = render_result["original_hp"]
                        # Prepare mask for evaluation (pass to healpix_ssim for proper neighborhood handling)
                        eval_mask = None
                        if healpix_mask is not None:
                            eval_mask = healpix_mask.to(dtype=image.dtype, device=image.device)
                        # Compute SSIM with proper mask support
                        current_ssim = healpix_ssim(image, gt_image, train=False, mask=eval_mask).item()
                        # Free HEALPix tensors immediately after SSIM computation
                        del image, gt_image
                        
                        # Project HEALPix/equirectangular render to original camera space for L1/PSNR
                        if perspective_params is not None:
                            # Perspective mode: project HEALPix to perspective
                            from scene.cameras import healpix_to_perspective
                            hp_render = render_result["hp_color"].detach()
                            image = healpix_to_perspective(
                                hp_render,
                                viewpoint.FoVx, viewpoint.FoVy,
                                perspective_params['W'], perspective_params['H']
                            )
                            image = torch.clamp(image, 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        elif fisheye_params is not None:
                            # Fisheye mode: project HEALPix render directly to fisheye
                            from scene.cameras import healpix_to_fisheye
                            hp_render = render_result["hp_color"].detach()
                            image = healpix_to_fisheye(
                                hp_render, 
                                fisheye_params['fx'], fisheye_params['fy'],
                                fisheye_params['cx'], fisheye_params['cy'],
                                fisheye_params['w_x'], fisheye_params['w_y'],
                                fisheye_params['W'], fisheye_params['H']
                            )
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        else:
                            # Omnidirectional (camera_model==2): render is equirectangular
                            image = torch.clamp(render_result["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    else:
                        image = torch.clamp(render_result["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if FUSED_SSIM_AVAILABLE:
                            current_ssim = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
                        else:
                            current_ssim = ssim(image, gt_image).item()
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += current_ssim
                    # Explicitly delete tensors and clear cache after each viewpoint
                    del image, gt_image, current_gt_image, render_result
                    torch.cuda.empty_cache()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_avg = ssim_test / len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.2f} SSIM {ssim_avg:.4f}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/ssim", ssim_avg, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--healpix_scale", type=int, default=0)
    parser.add_argument("--healpix_scale_test", type=int, default=None, help="HEALPix scale for testing and evaluation (required for all camera models).")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # All camera models render through HEALPix.
    if args.healpix_scale_test is None:
        parser.error("--healpix_scale_test is required (all camera models use HEALPix rendering)")
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.healpix_scale, args.healpix_scale_test)

    # All done
    print("\nTraining complete.")
