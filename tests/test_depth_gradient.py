"""
Test script to verify depth supervision gradient flow in HEALPix mode.
Run this after rebuilding the CUDA extension.

Usage:
    cd /path/to/UniTriSplat
    python tests/test_depth_gradient.py
"""

import torch

# Import from the installed package (make sure to pip install -e . first)
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def test_depth_gradient():
    device = torch.device("cuda:0")
    
    # Create simple test case: 2 Gaussians
    num_gaussians = 2
    
    # Gaussian parameters (requires_grad=True for trainable params)
    # Note: Camera looks along +Z, Y is up. Place Gaussians in front of camera.
    # Avoid placing directly on poles (lon=0, lat=0 corresponds to +X direction for equirectangular)
    # For equirectangular: lon=0 is +X, lon=pi/2 is +Z, lat=0 is equator
    means3D = torch.tensor([
        [2.0, 0.0, 0.1],   # Gaussian 1: mostly in +X direction, distance ~2.0
        [1.0, 0.5, 1.5],   # Gaussian 2: mix of directions, distance ~1.87
    ], device=device, dtype=torch.float32, requires_grad=True)
    
    # Simple spherical Gaussians - larger size to ensure they're visible
    scales = torch.tensor([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ], device=device, dtype=torch.float32, requires_grad=True)
    
    # Identity rotations
    rotations = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], device=device, dtype=torch.float32, requires_grad=True)
    
    # Opacities
    opacities = torch.tensor([
        [0.9],
        [0.9],
    ], device=device, dtype=torch.float32, requires_grad=True)
    
    # Spherical harmonics (just DC component for simplicity)
    shs = torch.ones(num_gaussians, 1, 3, device=device, dtype=torch.float32, requires_grad=True)
    
    # means2D is actually used to store screen-space gradients, shape must be {N, 3} like means3D
    means2D = torch.zeros(num_gaussians, 3, device=device, dtype=torch.float32, requires_grad=True)
    
    # Camera parameters: identity view matrix (camera at origin looking down +Z)
    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    projmatrix = torch.eye(4, device=device, dtype=torch.float32)  # Not used for omni
    campos = torch.zeros(3, device=device, dtype=torch.float32)
    bg = torch.zeros(3, device=device, dtype=torch.float32)
    
    # Small image (for testing)
    H, W = 64, 128  # Equirectangular dimensions
    
    # Create rasterization settings for omnidirectional mode (camera_model=2)
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,  # Not used for omni
        tanfovy=1.0,  # Not used for omni
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
        camera_model=2,  # Omnidirectional
        original_image=torch.zeros(3, H, W, device=device),
        healpix_scale=0,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Forward pass
    color, radii, radius_rad, invdepth, hp_color, hp_invdepth, original_hp = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        shs=shs,
        scales=scales,
        rotations=rotations,
    )
    
    print("="*60)
    print("Forward Pass Results:")
    print("="*60)
    print(f"hp_invdepth shape: {hp_invdepth.shape}")
    print(f"hp_invdepth requires_grad: {hp_invdepth.requires_grad}")
    print(f"hp_invdepth max: {hp_invdepth.max().item():.6f}")
    print(f"hp_invdepth mean (non-zero): {hp_invdepth[hp_invdepth > 0].mean().item():.6f}")
    
    # Expected inverse depths:
    # Gaussian 1: distance = sqrt(4 + 0 + 0.01) = 2.002, inv_depth = 0.499
    # Gaussian 2: distance = sqrt(1 + 0.25 + 2.25) = 1.87, inv_depth = 0.535
    d1 = torch.sqrt(torch.tensor(2.0**2 + 0.0**2 + 0.1**2))
    d2 = torch.sqrt(torch.tensor(1.0**2 + 0.5**2 + 1.5**2))
    print(f"\nExpected inv depths:")
    print(f"  Gaussian 1: 1/{d1.item():.3f} = {(1/d1).item():.4f}")
    print(f"  Gaussian 2: 1/{d2.item():.3f} = {(1/d2).item():.4f}")
    
    # Create fake ground truth depth (all pixels should have inv_depth = 0.4)
    hp_depth_gt = torch.full_like(hp_invdepth, 0.4)
    
    # Compute L1 depth loss
    depth_loss = torch.abs(hp_invdepth - hp_depth_gt).mean()
    
    print(f"\nDepth loss: {depth_loss.item():.6f}")
    
    # Backward pass
    depth_loss.backward()
    
    print("\n" + "="*60)
    print("Backward Pass Results - Gradient Check:")
    print("="*60)
    
    print(f"\nmeans3D.grad:")
    if means3D.grad is not None:
        print(f"  Shape: {means3D.grad.shape}")
        print(f"  Values:\n{means3D.grad}")
        print(f"  Grad norm: {means3D.grad.norm().item():.8f}")
    else:
        print("  None - NO GRADIENT FLOW!")
    
    print(f"\nscales.grad:")
    if scales.grad is not None:
        print(f"  Shape: {scales.grad.shape}")
        print(f"  Grad norm: {scales.grad.norm().item():.8f}")
    else:
        print("  None")
    
    print(f"\nrotations.grad:")
    if rotations.grad is not None:
        print(f"  Shape: {rotations.grad.shape}")
        print(f"  Grad norm: {rotations.grad.norm().item():.8f}")
    else:
        print("  None")
    
    print(f"\nopacities.grad:")
    if opacities.grad is not None:
        print(f"  Shape: {opacities.grad.shape}")
        print(f"  Grad norm: {opacities.grad.norm().item():.8f}")
    else:
        print("  None")
    
    # Verify gradient direction makes sense
    print("\n" + "="*60)
    print("Gradient Sanity Check:")
    print("="*60)
    
    if means3D.grad is not None and means3D.grad.norm().item() > 1e-10:
        print("\n✓ Gradients are flowing through depth supervision!")
        
        # Check both Gaussians
        for i in range(num_gaussians):
            pos = means3D[i].detach()
            grad = means3D.grad[i]
            dist = torch.sqrt((pos**2).sum()).item()
            inv_d = 1.0 / dist
            
            print(f"\nGaussian {i+1} analysis:")
            print(f"  Position: ({pos[0].item():.3f}, {pos[1].item():.3f}, {pos[2].item():.3f})")
            print(f"  Distance: {dist:.4f}, inv_depth: {inv_d:.4f}")
            print(f"  GT inv_depth: 0.4")
            print(f"  Gradient: ({grad[0].item():.6f}, {grad[1].item():.6f}, {grad[2].item():.6f})")
            print(f"  Gradient norm: {grad.norm().item():.8f}")
            
            # Check if gradient direction is correct
            # If inv_depth > GT (0.4), we need to increase depth (move away from camera)
            # The gradient of loss w.r.t. position should point toward camera (negative direction)
            # So that pos -= lr*grad moves the Gaussian away
            if inv_d > 0.4:
                print(f"  inv_depth > GT: gradient should point TOWARD camera")
                # Gradient dot position should be positive (gradient points toward origin)
                dot_product = (grad * pos).sum().item()
                if dot_product > 0:
                    print(f"  ✓ grad·pos = {dot_product:.6f} > 0, CORRECT direction")
                elif abs(dot_product) < 1e-8:
                    print(f"  ? grad·pos = {dot_product:.6f} ≈ 0, gradient too small to verify")
                else:
                    print(f"  ✗ grad·pos = {dot_product:.6f} < 0, WRONG direction")
    else:
        print("\n✗ NO gradients flowing through depth supervision!")
        print("  Please rebuild the CUDA extension:")
        print("  cd submodules/diff-gaussian-rasterization && pip install -e .")

if __name__ == "__main__":
    test_depth_gradient()
