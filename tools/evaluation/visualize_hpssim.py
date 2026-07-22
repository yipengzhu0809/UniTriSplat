#!/usr/bin/env python3
"""
Visualize HEALPix SSIM map for pairs of GT and rendered images.

Usage:
    python visualize_hpssim.py --input_dir <folder> [--output_dir <output_folder>] [--nside <N_side>]

The input folder should contain:
    - gt/       : Ground truth images
    - renders/  : Rendered images

Each pair of images with the same filename will be compared.
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    print("[WARNING] healpy not available, mollview visualization will be skipped")
    HEALPY_AVAILABLE = False

try:
    from healpix_ssim import healpix_ssim, healpix_ssim_map
    HEALPIX_SSIM_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] healpix_ssim not available: {e}")
    print("Please install healpix_ssim: cd submodules/healpix-ssim && pip install -e .")
    HEALPIX_SSIM_AVAILABLE = False


def equirectangular_to_healpix(image: torch.Tensor, N_side: int) -> torch.Tensor:
    """
    Convert equirectangular image to HEALPix format (NESTED order).
    
    Args:
        image: (C, H, W) equirectangular image tensor
        N_side: HEALPix N_side parameter
    
    Returns:
        hp_image: (C, N_pix) HEALPix image tensor
    """
    C, H, W = image.shape
    N_pix = 12 * N_side * N_side
    
    # Generate HEALPix pixel coordinates
    pix_indices = np.arange(N_pix)
    # Convert NESTED to RING for healpy, get angles, then we'll work in NESTED
    theta, phi = hp.pix2ang(N_side, pix_indices, nest=True)
    
    # Convert to equirectangular coordinates
    # theta: [0, pi] -> v: [0, H-1]
    # phi: [0, 2*pi] -> u: [0, W-1]
    # Note: equirectangular center is phi=0, so phi in [-pi, pi] maps to u in [0, W-1]
    # phi from healpy is in [0, 2*pi], convert to [-pi, pi] then to u
    phi_centered = ((phi + np.pi) % (2 * np.pi)) - np.pi  # shift to [-pi, pi]
    v = theta / np.pi * (H - 1)
    u = (phi_centered / (2 * np.pi) + 0.5) * (W - 1)  # [-pi, pi] -> [0, W-1]
    
    # Bilinear interpolation
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    u0 = np.clip(u0, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    
    du = u - u0
    dv = v - v0
    
    du = torch.from_numpy(du).float().to(image.device)
    dv = torch.from_numpy(dv).float().to(image.device)
    
    # Gather pixel values
    image_np = image.cpu().numpy()
    
    hp_image = np.zeros((C, N_pix), dtype=np.float32)
    for c in range(C):
        img_c = image_np[c]
        val00 = img_c[v0, u0]
        val01 = img_c[v0, u1]
        val10 = img_c[v1, u0]
        val11 = img_c[v1, u1]
        
        du_np = du.cpu().numpy()
        dv_np = dv.cpu().numpy()
        
        hp_image[c] = (val00 * (1 - du_np) * (1 - dv_np) +
                       val01 * du_np * (1 - dv_np) +
                       val10 * (1 - du_np) * dv_np +
                       val11 * du_np * dv_np)
    
    return torch.from_numpy(hp_image).to(image.device)


def healpix_to_equirectangular(hp_image: torch.Tensor, W: int, H: int, flip: bool = False, interpolate: bool = True) -> torch.Tensor:
    """
    Convert HEALPix image (NESTED order) to equirectangular format.
    
    Args:
        hp_image: (C, N_pix) HEALPix image tensor
        W: output width
        H: output height
        flip: if True, shift longitude by 180 degrees
        interpolate: if True, use bilinear interpolation (default True for better quality)
    
    Returns:
        equi_image: (C, H, W) equirectangular image tensor
    """
    C, N_pix = hp_image.shape
    N_side = int(np.sqrt(N_pix / 12))
    
    # Generate equirectangular pixel coordinates
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Convert to spherical coordinates
    # theta: [0, pi] (colatitude)
    # phi: [-pi, pi] -> center of image is phi=0, matching mollview convention
    theta = v / (H - 1) * np.pi  # [0, pi]
    phi = (u / (W - 1) - 0.5) * 2 * np.pi  # [-pi, pi], center at phi=0
    phi = phi % (2 * np.pi)  # wrap to [0, 2*pi] for healpy
    
    # Flip longitude by 180 degrees if requested
    if flip:
        phi = (phi + np.pi) % (2 * np.pi)
    
    hp_np = hp_image.cpu().numpy()
    equi_image = np.zeros((C, H, W), dtype=np.float32)
    
    if interpolate:
        # Bilinear interpolation for smoother results
        pix_ids, weights = hp.get_interp_weights(N_side, theta.flatten(), phi.flatten(), nest=True)
        for c in range(C):
            interp_val = np.sum(hp_np[c, pix_ids] * weights, axis=0)
            equi_image[c] = interp_val.reshape(H, W)
    else:
        # Nearest neighbor sampling (original method)
        pix_indices = hp.ang2pix(N_side, theta.flatten(), phi.flatten(), nest=True)
        for c in range(C):
            equi_image[c] = hp_np[c, pix_indices].reshape(H, W)
    
    return torch.from_numpy(equi_image).to(hp_image.device)


def load_image(path: str) -> torch.Tensor:
    """Load image and convert to (C, H, W) tensor in [0, 1]."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return img_tensor


def visualize_ssim_map(hp_ssim_map: np.ndarray, output_path: str, title: str, 
                       W: int = 2048, H: int = 1024, flip: bool = False):
    """
    Visualize SSIM map in both equirectangular and mollview formats.
    
    Args:
        hp_ssim_map: (N_pix,) HEALPix SSIM map in NESTED order
        output_path: output path prefix (without extension)
        title: title for the plot
        W, H: equirectangular output dimensions
        flip: if True, shift longitude by 180 degrees
    """
    N_pix = hp_ssim_map.shape[0]
    N_side = int(np.sqrt(N_pix / 12))
    
    # Convert to equirectangular
    hp_tensor = torch.from_numpy(hp_ssim_map).unsqueeze(0).float()  # (1, N_pix)
    equi_ssim = healpix_to_equirectangular(hp_tensor, W, H, flip=flip).numpy()[0]  # (H, W)
    
    mean_ssim = hp_ssim_map.mean()
    
    # Save equirectangular SSIM map
    plt.figure(figsize=(14, 7))
    im = plt.imshow(equi_ssim, cmap='RdYlGn', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='SSIM', shrink=0.8)
    plt.title(f"{title}\nMean SSIM: {mean_ssim:.4f}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_path}_equi.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save equirectangular error map
    error_map = 1.0 - equi_ssim
    plt.figure(figsize=(14, 7))
    im = plt.imshow(error_map, cmap='hot', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='1 - SSIM (Error)', shrink=0.8)
    plt.title(f"{title} - Error Map\nMean Error: {1-mean_ssim:.4f}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_path}_error_equi.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save mollview (spherical projection)
    if HEALPY_AVAILABLE:
        # Convert NESTED to RING for healpy visualization
        hp_ssim_ring = hp.reorder(hp_ssim_map, n2r=True)
        
        # Rotate to match equirectangular coordinate system
        # flip adds 180 degrees rotation
        rot_angle = [180 if flip else 0, 0, 0]
        
        plt.figure(figsize=(12, 7))
        # Use flip='geo' to match equirectangular left-right orientation
        hp.mollview(hp_ssim_ring, title=f"{title} (Mean: {mean_ssim:.4f})",
                   cmap='RdYlGn', min=0.0, max=1.0, hold=True, rot=rot_angle, flip='geo')
        plt.savefig(f"{output_path}_mollview.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save mollview error map
        hp_error_ring = 1.0 - hp_ssim_ring
        plt.figure(figsize=(12, 7))
        hp.mollview(hp_error_ring, title=f"{title} - Error (Mean: {1-mean_ssim:.4f})",
                   cmap='hot', min=0.0, max=1.0, hold=True, rot=rot_angle, flip='geo')
        plt.savefig(f"{output_path}_error_mollview.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    return mean_ssim


def process_image_pair(gt_path: str, render_path: str, output_dir: str, 
                       N_side: int, device: str = 'cuda', flip: bool = False):
    """
    Process a pair of GT and rendered images.
    
    Args:
        flip: if True, shift longitude by 180 degrees in visualization
    
    Returns:
        mean_ssim: mean SSIM value
    """
    # Load images
    gt_img = load_image(gt_path).to(device)
    render_img = load_image(render_path).to(device)
    
    # Check dimensions match
    if gt_img.shape != render_img.shape:
        print(f"[WARNING] Shape mismatch: GT {gt_img.shape} vs Render {render_img.shape}")
        # Resize render to match GT
        from torchvision.transforms.functional import resize
        render_img = resize(render_img, [gt_img.shape[1], gt_img.shape[2]])
    
    # Convert to HEALPix
    hp_gt = equirectangular_to_healpix(gt_img, N_side)  # (C, N_pix)
    hp_render = equirectangular_to_healpix(render_img, N_side)  # (C, N_pix)
    
    # Compute SSIM map
    with torch.no_grad():
        hp_ssim = healpix_ssim_map(hp_render, hp_gt, train=False)  # (C, N_pix)
        # Average across channels
        hp_ssim_avg = hp_ssim.mean(dim=0)  # (N_pix,)
    
    # Get base filename
    basename = os.path.splitext(os.path.basename(gt_path))[0]
    output_path = os.path.join(output_dir, basename)
    
    # Visualize
    hp_ssim_np = hp_ssim_avg.cpu().numpy()
    W, H = gt_img.shape[2], gt_img.shape[1]
    mean_ssim = visualize_ssim_map(hp_ssim_np, output_path, f"HEALPix SSIM: {basename}", W, H, flip=flip)
    
    return mean_ssim


def main():
    parser = argparse.ArgumentParser(description="Visualize HEALPix SSIM maps for GT/render pairs")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing 'gt' and 'renders' subfolders")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: input_dir/hpssim_viz)")
    parser.add_argument("--nside", type=int, default=512,
                       help="HEALPix N_side parameter (default: 512)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--ext", type=str, default="png",
                       help="Image extension to search for (default: png)")
    parser.add_argument("--flip", action="store_true",
                       help="Flip longitude by 180 degrees in visualization")
    args = parser.parse_args()
    
    if not HEALPIX_SSIM_AVAILABLE:
        print("[ERROR] healpix_ssim module is required. Please install it first.")
        return
    
    # Setup paths
    gt_dir = os.path.join(args.input_dir, "gt")
    render_dir = os.path.join(args.input_dir, "renders")
    
    if not os.path.exists(gt_dir):
        print(f"[ERROR] GT directory not found: {gt_dir}")
        return
    if not os.path.exists(render_dir):
        print(f"[ERROR] Renders directory not found: {render_dir}")
        return
    
    output_dir = args.output_dir or os.path.join(args.input_dir, "hpssim_viz")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image pairs
    gt_images = sorted(glob(os.path.join(gt_dir, f"*.{args.ext}")))
    if not gt_images:
        # Try other extensions
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            gt_images = sorted(glob(os.path.join(gt_dir, f"*.{ext}")))
            if gt_images:
                args.ext = ext
                break
    
    if not gt_images:
        print(f"[ERROR] No images found in {gt_dir}")
        return
    
    print(f"Found {len(gt_images)} GT images")
    print(f"Using N_side = {args.nside} (N_pix = {12 * args.nside * args.nside})")
    print(f"Output directory: {output_dir}")
    
    # Process each pair
    ssim_values = []
    for gt_path in tqdm(gt_images, desc="Processing image pairs"):
        basename = os.path.basename(gt_path)
        render_path = os.path.join(render_dir, basename)
        
        if not os.path.exists(render_path):
            print(f"[WARNING] Render not found for {basename}, skipping")
            continue
        
        try:
            mean_ssim = process_image_pair(gt_path, render_path, output_dir, 
                                           args.nside, args.device, args.flip)
            ssim_values.append((basename, mean_ssim))
            print(f"  {basename}: SSIM = {mean_ssim:.4f}")
        except Exception as e:
            print(f"[ERROR] Failed to process {basename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if ssim_values:
        avg_ssim = np.mean([s[1] for s in ssim_values])
        print(f"\n{'='*50}")
        print(f"Processed {len(ssim_values)} image pairs")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Output saved to: {output_dir}")
        
        # Save summary to file
        summary_path = os.path.join(output_dir, "ssim_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"HEALPix SSIM Summary\n")
            f.write(f"N_side: {args.nside}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
            f.write(f"Per-image SSIM:\n")
            for name, ssim in sorted(ssim_values, key=lambda x: x[1]):
                f.write(f"  {name}: {ssim:.4f}\n")
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
