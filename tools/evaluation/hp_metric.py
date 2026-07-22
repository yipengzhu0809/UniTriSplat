#!/usr/bin/env python3
"""
Compute HEALPix metrics for corresponding images in renders and gt folders.

The script converts each equirectangular image pair to NESTED HEALPix order,
computes the requested metrics, and reports dataset averages.
"""

import numpy as np
import os
import argparse
import glob
from tqdm import tqdm
import cv2
import torch

try:
    import healpy as hp
except ImportError:
    print("Please install healpy: pip install healpy")
    exit(1)

from healpix_ssim import healpix_ssim


def equirectangular_to_healpix_color(image, nside):
    """
    Convert an RGB equirectangular image to a NESTED HEALPix map.
    
    Args:
        image: RGB float32 array with shape (H, W, 3) and values in [0, 1].
        nside: HEALPix NSIDE value.
    
    Returns:
        HEALPix color array with shape (Npix, 3).
    """
    H, W = image.shape[:2]
    npix = hp.nside2npix(nside)  # 12 * nside^2
    
    # Obtain angular coordinates for all HEALPix pixels.
    pix_indices = np.arange(npix)
    
    # Obtain theta and phi in NESTED order.
    theta, phi = hp.pix2ang(nside, pix_indices, nest=True)
    
    # Convert to latitude and longitude using the CUDA convention.
    # CUDA: pix_lat = π/2 - theta, pix_lon = phi - π
    lat = np.pi / 2 - theta
    lon = phi - np.pi
    
    # Convert longitude and latitude to equirectangular pixels.
    # Match the CUDA lonlat2Equipixel_index mapping:
    u = (lon / np.pi + 1.0) * 0.5  # (lon + π) / (2π)
    v = 0.5 + lat / np.pi          # (π/2 + lat) / π
    
    x_f = u * W - 0.5
    y_f = v * H - 0.5
    
    # Round to nearest integer (equivalent to floor(x + 0.5))
    xi = np.floor(x_f + 0.5).astype(int)
    yi = np.floor(y_f + 0.5).astype(int)
    
    # Longitude wrapping (periodic)
    xi = np.where(xi < 0, (xi % W + W) % W, xi)
    xi = np.where(xi >= W, xi % W, xi)
    
    # Latitude clamping
    yi = np.clip(yi, 0, H - 1)
    
    # Sample RGB image values.
    healpix_map = image[yi, xi]  # npix x 3
    
    return healpix_map


def healpix_mse(hp_render, hp_gt):
    """
    Compute mean squared error between two HEALPix maps.
    """
    return np.mean((hp_render - hp_gt) ** 2)


def healpix_psnr(hp_render, hp_gt):
    """
    Compute PSNR between two HEALPix maps.
    """
    mse = healpix_mse(hp_render, hp_gt)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser(description='Compute HEALPix SSIM for image pairs in renders and gt folders')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Input directory containing renders and gt subfolders')
    parser.add_argument('--nside', '-n', type=int, default=512,
                        help='HEALPix NSIDE value (default: 512)')
    parser.add_argument('--order', '-o', type=int, default=None,
                        help='HEALPix order (nside=2^order); overrides --nside')
    parser.add_argument('--ext', type=str, default='png',
                        help='Image extension (default: png)')
    
    args = parser.parse_args()
    
    # Resolve NSIDE.
    if args.order is not None:
        nside = 1 << args.order
    else:
        nside = args.nside
    
    # Verify that NSIDE is a power of two.
    if nside & (nside - 1) != 0:
        print(f"Error: nside must be a power of 2, got {nside}")
        return
    
    order = int(np.log2(nside))
    npix = 12 * nside * nside
    
    print("=" * 60)
    print("HEALPix SSIM Calculator")
    print("=" * 60)
    print(f"\nInput directory: {args.input_dir}")
    print(f"HEALPix params: order={order}, nside={nside}, npix={npix}")
    
    # Validate input folders.
    renders_dir = os.path.join(args.input_dir, 'renders')
    gt_dir = os.path.join(args.input_dir, 'gt')
    
    if not os.path.exists(renders_dir):
        print(f"Error: renders folder not found: {renders_dir}")
        return
    if not os.path.exists(gt_dir):
        print(f"Error: gt folder not found: {gt_dir}")
        return
    
    # Collect image files.
    render_files = sorted(glob.glob(os.path.join(renders_dir, f'*.{args.ext}')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, f'*.{args.ext}')))
    
    if len(render_files) == 0:
        print(f"Error: No {args.ext} files found in renders folder")
        return
    if len(gt_files) == 0:
        print(f"Error: No {args.ext} files found in gt folder")
        return
    
    # Match files by name.
    render_dict = {os.path.basename(f): f for f in render_files}
    gt_dict = {os.path.basename(f): f for f in gt_files}
    
    common_names = set(render_dict.keys()) & set(gt_dict.keys())
    if len(common_names) == 0:
        print("Error: No matching image pairs found")
        print(f"  Render files: {list(render_dict.keys())[:5]}...")
        print(f"  GT files: {list(gt_dict.keys())[:5]}...")
        return
    
    common_names = sorted(common_names)
    print(f"\nFound {len(common_names)} matching image pairs")
    
    # Compute HEALPix metrics for each image pair.
    print("\nProcessing images...")
    
    ssim_values = []
    psnr_values = []
    mse_values = []
    
    for name in tqdm(common_names, desc="Computing HEALPix metrics"):
        render_path = render_dict[name]
        gt_path = gt_dict[name]
        
        # Read the image, convert BGR to RGB, and normalize to [0, 1].
        render_img = cv2.imread(render_path)
        gt_img = cv2.imread(gt_path)
        
        if render_img is None:
            print(f"  Warning: Cannot read {render_path}")
            continue
        if gt_img is None:
            print(f"  Warning: Cannot read {gt_path}")
            continue
        
        # Convert BGR to RGB and normalize.
        render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Require matching image dimensions.
        if render_img.shape != gt_img.shape:
            print(f"  Warning: Size mismatch for {name}: {render_img.shape} vs {gt_img.shape}")
            continue
        
        # Convert to HEALPix.
        hp_render = equirectangular_to_healpix_color(render_img, nside)
        hp_gt = equirectangular_to_healpix_color(gt_img, nside)
        
        # Compute metrics.
        # healpix_ssim expects a torch.Tensor with shape (3, Npix).
        # equirectangular_to_healpix_color returns (Npix, 3), so transpose it.
        hp_render_tensor = torch.from_numpy(hp_render).T.contiguous().cuda()
        hp_gt_tensor = torch.from_numpy(hp_gt).T.contiguous().cuda()
        ssim = healpix_ssim(hp_render_tensor, hp_gt_tensor).item()
        psnr = healpix_psnr(hp_render, hp_gt)
        mse = healpix_mse(hp_render, hp_gt)
        
        ssim_values.append(ssim)
        psnr_values.append(psnr)
        mse_values.append(mse)
    
    # Compute dataset averages.
    if len(ssim_values) == 0:
        print("\nError: No valid image pairs processed")
        return
    
    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)
    mean_mse = np.mean(mse_values)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nProcessed {len(ssim_values)} image pairs")
    print(f"\n{'Metric':<20} {'Mean':>15} {'Std':>15} {'Min':>15} {'Max':>15}")
    print("-" * 80)
    print(f"{'HEALPix SSIM':<20} {mean_ssim:>15.6f} {np.std(ssim_values):>15.6f} {np.min(ssim_values):>15.6f} {np.max(ssim_values):>15.6f}")
    print(f"{'HEALPix PSNR (dB)':<20} {mean_psnr:>15.4f} {np.std(psnr_values):>15.4f} {np.min(psnr_values):>15.4f} {np.max(psnr_values):>15.4f}")
    print(f"{'HEALPix MSE':<20} {mean_mse:>15.6f} {np.std(mse_values):>15.6f} {np.min(mse_values):>15.6f} {np.max(mse_values):>15.6f}")
    
    # Save results.
    results = {
        'num_images': len(ssim_values),
        'nside': nside,
        'order': order,
        'npix': npix,
        'healpix_ssim': {
            'mean': float(mean_ssim),
            'std': float(np.std(ssim_values)),
            'min': float(np.min(ssim_values)),
            'max': float(np.max(ssim_values)),
            'values': [float(v) for v in ssim_values]
        },
        'healpix_psnr': {
            'mean': float(mean_psnr),
            'std': float(np.std(psnr_values)),
            'min': float(np.min(psnr_values)),
            'max': float(np.max(psnr_values)),
            'values': [float(v) for v in psnr_values]
        },
        'healpix_mse': {
            'mean': float(mean_mse),
            'std': float(np.std(mse_values)),
            'min': float(np.min(mse_values)),
            'max': float(np.max(mse_values)),
            'values': [float(v) for v in mse_values]
        },
        'image_names': common_names
    }
    
    output_path = os.path.join(args.input_dir, 'healpix_metrics.json')
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

