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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False


def equirectangular_to_healpix_gt(image, nside):
    """
    Convert an equirectangular image to a NESTED HEALPix map.
    
    Args:
        image: Image tensor with shape (C, H, W), float32 values in [0, 1].
        nside: HEALPix NSIDE value; it must be a power of two.
    
    Returns:
        HEALPix image tensor with shape (C, Npix).
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required for HEALPix precomputation. Install with: pip install healpy")
    
    C, H, W = image.shape
    npix = 12 * nside * nside
    order = int(np.log2(nside))
    
    device = image.device
    
    image_np = image.cpu().numpy()
    
    pix_indices = np.arange(npix)
    
    theta, phi = hp.pix2ang(nside, pix_indices, nest=True)
    
    pix_lat = np.pi / 2.0 - theta  # latitude: [-π/2, π/2]
    pix_lon = phi - np.pi           # longitude: [-π, π)
    
    pix_lon = np.where(pix_lon < -np.pi, pix_lon + 2 * np.pi, pix_lon)
    pix_lon = np.where(pix_lon >= np.pi, pix_lon - 2 * np.pi, pix_lon)
    
    u = (pix_lon / np.pi + 1.0) * 0.5  # [0, 1)
    v = 0.5 + pix_lat / np.pi          # [0, 1]
    
    x_f = u * W - 0.5
    y_f = v * H - 0.5
    
    xi = np.floor(x_f + 0.5).astype(np.int32)
    yi = np.floor(y_f + 0.5).astype(np.int32)
    
    xi = np.where(xi < 0, (xi % W + W) % W, xi)
    xi = np.where(xi >= W, xi % W, xi)
    
    yi = np.clip(yi, 0, H - 1)
    
    healpix_map = np.zeros((C, npix), dtype=np.float32)
    for ch in range(C):
        healpix_map[ch] = image_np[ch, yi, xi]
    
    return torch.from_numpy(healpix_map).to(device)


def equirectangular_depth_to_healpix(invdepthmap, nside):
    """
    Convert an equirectangular inverse-depth map to NESTED HEALPix order.
    
    Args:
        invdepthmap: Inverse-depth tensor with shape (1, H, W).
        nside: HEALPix NSIDE value; it must be a power of two.
    
    Returns:
        HEALPix inverse-depth tensor with shape (1, Npix).
    """
    return equirectangular_to_healpix_gt(invdepthmap, nside)


def perspective_to_healpix_gt(image, FoVx, FoVy, nside):
    """
    Project a perspective image onto a NESTED HEALPix grid.
    
    Each HEALPix direction is projected through the pinhole camera model and
    sampled when it lies inside the camera frustum. Camera coordinates use +Z
    forward, +X right, and +Y down.
    
    Args:
        image: Perspective image tensor with shape (C, H, W).
        FoVx: Horizontal field of view in radians.
        FoVy: Vertical field of view in radians.
        nside: HEALPix NSIDE value; it must be a power of two.
    
    Returns:
        A tuple containing the (C, Npix) HEALPix image and the (Npix,) validity mask.
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required for HEALPix precomputation. Install with: pip install healpy")
    
    C, H, W = image.shape
    npix = 12 * nside * nside
    device = image.device
    image_np = image.cpu().numpy()
    
    pix_indices = np.arange(npix)
    
    # HEALPix pixel -> spherical coordinates (theta, phi)
    theta, phi = hp.pix2ang(nside, pix_indices, nest=True)
    
    # Convert to 3D directions in camera coordinates.
    # Keep the convention consistent with fisheye_to_healpix_gt and CUDA p3d2lonlat.
    X = np.sin(theta) * np.sin(phi - np.pi)  # Shift azimuth by pi so phi=pi points forward.
    Y = np.cos(theta)  # Y = cos(theta)
    Z = np.sin(theta) * np.cos(phi - np.pi)  # +Z points forward.
    
    # A sample is valid only when it is in front of the camera and inside the frustum.
    valid_mask = Z > 0
    
    # Initialize outputs.
    healpix_map = np.zeros((C, npix), dtype=np.float32)
    
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return torch.from_numpy(healpix_map).to(device), torch.from_numpy(valid_mask).to(device)
    
    X_valid = X[valid_indices]
    Y_valid = Y[valid_indices]
    Z_valid = Z[valid_indices]
    
    # Normalize to unit directions.
    norm = np.sqrt(X_valid**2 + Y_valid**2 + Z_valid**2)
    X_valid = X_valid / norm
    Y_valid = Y_valid / norm
    Z_valid = Z_valid / norm
    
    # Pinhole projection: 3D direction -> pixel coordinates.
    # Match the CUDA preprocessCUDA convention:
    #   point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) }
    #   ndc2Pix(v, S) = ((v + 1.0) * S - 1.0) * 0.5
    
    tan_fovx = np.tan(FoVx * 0.5)
    tan_fovy = np.tan(FoVy * 0.5)
    
    # Project to normalized device coordinates (NDC).
    # x_ndc = (X/Z) / tan_fovx, y_ndc = (Y/Z) / tan_fovy
    x_ndc = (X_valid / Z_valid) / tan_fovx
    y_ndc = (Y_valid / Z_valid) / tan_fovy
    
    # Test the frustum bounds in the NDC range [-1, 1].
    in_frustum = (np.abs(x_ndc) <= 1.0) & (np.abs(y_ndc) <= 1.0)
    
    # Convert to pixel coordinates using the CUDA ndc2Pix convention.
    # ndc2Pix(v, S) = ((v + 1.0) * S - 1.0) * 0.5
    u = ((x_ndc + 1.0) * W - 1.0) * 0.5
    v = ((y_ndc + 1.0) * H - 1.0) * 0.5
    
    # Check image bounds.
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    final_valid_relative = in_frustum & in_bounds
    
    # Update the validity mask.
    valid_mask_updated = np.zeros(npix, dtype=bool)
    valid_mask_updated[valid_indices[final_valid_relative]] = True
    
    # Sample colors with nearest-neighbor lookup.
    final_valid_indices = valid_indices[final_valid_relative]
    u_final = np.round(u[final_valid_relative]).astype(np.int32)
    v_final = np.round(v[final_valid_relative]).astype(np.int32)
    
    # Clamp indices to prevent out-of-bounds access.
    u_final = np.clip(u_final, 0, W - 1)
    v_final = np.clip(v_final, 0, H - 1)
    
    for ch in range(C):
        healpix_map[ch, final_valid_indices] = image_np[ch, v_final, u_final]
    
    return torch.from_numpy(healpix_map).to(device), torch.from_numpy(valid_mask_updated).to(device)


def generate_perspective_healpix_mask(FoVx, FoVy, width, height, nside):
    """
    Build a HEALPix visibility mask for a perspective camera.
    
    Args:
        FoVx: Horizontal field of view in radians.
        FoVy: Vertical field of view in radians.
        width: Perspective image width.
        height: Perspective image height.
        nside: HEALPix NSIDE value.
    
    Returns:
        Boolean tensor of shape (Npix,), true inside the camera frustum.
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required for HEALPix precomputation. Install with: pip install healpy")
    
    npix = 12 * nside * nside
    pix_indices = np.arange(npix)
    
    # HEALPix pixel -> spherical coordinates (theta, phi)
    theta, phi = hp.pix2ang(nside, pix_indices, nest=True)
    
    # Convert to 3D directions using the perspective_to_healpix_gt convention.
    X = np.sin(theta) * np.sin(phi - np.pi)
    Y = np.cos(theta)
    Z = np.sin(theta) * np.cos(phi - np.pi)
    
    # Only directions with Z > 0 can be projected.
    valid_mask = Z > 0
    
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return torch.zeros(npix, dtype=torch.bool)
    
    X_valid = X[valid_indices]
    Y_valid = Y[valid_indices]
    Z_valid = Z[valid_indices]
    
    # Normalize the directions.
    norm = np.sqrt(X_valid**2 + Y_valid**2 + Z_valid**2)
    X_valid = X_valid / norm
    Y_valid = Y_valid / norm
    Z_valid = Z_valid / norm
    
    # Apply the pinhole projection.
    tan_fovx = np.tan(FoVx * 0.5)
    tan_fovy = np.tan(FoVy * 0.5)
    
    x_ndc = (X_valid / Z_valid) / tan_fovx
    y_ndc = (Y_valid / Z_valid) / tan_fovy
    
    # Test the frustum bounds in the NDC range [-1, 1].
    in_frustum = (np.abs(x_ndc) <= 1.0) & (np.abs(y_ndc) <= 1.0)
    
    mask = np.zeros(npix, dtype=bool)
    mask[valid_indices[in_frustum]] = True
    
    return torch.from_numpy(mask)


def perspective_depth_to_healpix(invdepthmap, FoVx, FoVy, nside):
    """
    Project a perspective inverse-depth map onto a NESTED HEALPix grid.
    
    Returns the HEALPix inverse-depth tensor and its validity mask.
    """
    return perspective_to_healpix_gt(invdepthmap, FoVx, FoVy, nside)


def healpix_to_perspective(healpix_map, FoVx, FoVy, width, height, use_bilinear=True):
    """
    Synthesize a perspective image from a NESTED HEALPix map.
    
    Args:
        healpix_map: HEALPix image tensor with shape (C, Npix).
        FoVx: Horizontal field of view in radians.
        FoVy: Vertical field of view in radians.
        width: Output image width.
        height: Output image height.
        use_bilinear: Use HEALPix interpolation instead of nearest-neighbor sampling.
    
    Returns:
        Perspective image tensor with shape (C, height, width).
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required. Install with: pip install healpy")
    
    C, npix = healpix_map.shape
    nside = int(np.sqrt(npix / 12))
    device = healpix_map.device
    healpix_np = healpix_map.cpu().numpy()
    
    # Build the perspective pixel grid.
    u_coords = np.arange(width)
    v_coords = np.arange(height)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    u_flat = u_grid.flatten().astype(np.float64)
    v_flat = v_grid.flatten().astype(np.float64)
    
    # Pixel coordinates -> NDC coordinates (inverse ndc2Pix).
    # ndc2Pix(v, S) = ((v + 1.0) * S - 1.0) * 0.5
    # Inverse mapping: ndc = (2 * pix + 1) / S - 1.
    x_ndc = (2.0 * u_flat + 1.0) / width - 1.0
    y_ndc = (2.0 * v_flat + 1.0) / height - 1.0
    
    # NDC coordinates -> 3D directions in camera coordinates.
    tan_fovx = np.tan(FoVx * 0.5)
    tan_fovy = np.tan(FoVy * 0.5)
    
    X = x_ndc * tan_fovx
    Y = y_ndc * tan_fovy
    Z = np.ones_like(X)
    
    # Normalize the directions.
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X = X / norm
    Y = Y / norm
    Z = Z / norm
    
    # Convert to the HEALPix coordinate system.
    # Invert the transform used by perspective_to_healpix_gt:
    # Forward transform: X=sin(theta)sin(phi-pi), Y=cos(theta), Z=sin(theta)cos(phi-pi).
    
    # theta = arccos(Y)
    theta = np.arccos(np.clip(Y, -1.0, 1.0))
    
    # phi = arctan2(X, Z) + π
    phi = np.arctan2(X, Z) + np.pi
    
    # Wrap phi to [0, 2pi).
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    phi = np.where(phi >= 2 * np.pi, phi - 2 * np.pi, phi)
    
    perspective_image = np.zeros((C, height * width), dtype=np.float32)
    
    if use_bilinear:
        # Use healpy interpolation to reduce diamond-shaped aliasing.
        # hp.get_interp_val expects a RING-ordered map.
        for ch in range(C):
            # Convert NESTED order to RING order for interpolation.
            healpix_ring = hp.reorder(healpix_np[ch], n2r=True)
            # Interpolate the HEALPix samples.
            perspective_image[ch] = hp.get_interp_val(healpix_ring, theta, phi)
    else:
        # Nearest-neighbor sampling is faster but introduces aliasing.
        pix_indices = hp.ang2pix(nside, theta, phi, nest=True)
        for ch in range(C):
            perspective_image[ch] = healpix_np[ch, pix_indices]
    
    return torch.from_numpy(perspective_image.reshape(C, height, width)).to(device)


def healpix_to_equirectangular(healpix_map, W, H, interpolate=True):
    """
    Convert a NESTED HEALPix map to an equirectangular image.
    
    Args:
        healpix_map: HEALPix tensor with shape (C, Npix).
        W: Output image width.
        H: Output image height.
        interpolate: Use HEALPix interpolation when true.
    
    Returns:
        Equirectangular image tensor with shape (C, H, W).
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required. Install with: pip install healpy")
    
    C, npix = healpix_map.shape
    nside = int(np.sqrt(npix / 12))
    device = healpix_map.device
    healpix_np = healpix_map.cpu().numpy()
    
    # Build the equirectangular pixel grid.
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)  # (H, W)
    
    # Pixel centers -> normalized coordinates in [0, 1).
    # Match CUDA equipixel2Lonlat_center.
    u = (x_grid.astype(np.float64) + 0.5) / W  # [0, 1)
    v = (y_grid.astype(np.float64) + 0.5) / H  # [0, 1)
    
    # Normalized coordinates -> longitude and latitude.
    # Match CUDA: lon=(2u-1)pi and lat=-(0.5-v)pi.
    lon = (u * 2.0 - 1.0) * np.pi  # [-π, π)
    lat = -(0.5 - v) * np.pi       # lat = (v - 0.5) * π
    
    # lon/lat -> theta/phi (HEALPix convention)
    # theta ∈ [0, π], phi ∈ [0, 2π)
    theta = np.pi / 2.0 - lat  # θ = π/2 - lat
    phi = lon + np.pi          # φ = lon + π
    
    # Handle image boundaries.
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    phi = np.where(phi >= 2 * np.pi, phi - 2 * np.pi, phi)
    
    equi_image = np.zeros((C, H, W), dtype=np.float32)
    
    if interpolate:
        # Obtain the four neighboring HEALPix samples and their weights.
        pix_ids, weights = hp.get_interp_weights(nside, theta.flatten(), phi.flatten(), nest=True)
        # pix_ids: (4, H*W), weights: (4, H*W)
        
        for ch in range(C):
            # Compute the weighted interpolated value.
            interp_val = np.sum(healpix_np[ch, pix_ids] * weights, axis=0)
            equi_image[ch] = interp_val.reshape(H, W)
    else:
        # Use nearest-neighbor sampling.
        nest_ids = hp.ang2pix(nside, theta.flatten(), phi.flatten(), nest=True)
        nest_ids = nest_ids.reshape(H, W)
        
        for ch in range(C):
            equi_image[ch] = healpix_np[ch, nest_ids]
    
    return torch.from_numpy(equi_image).to(device)


def fisheye_to_healpix_gt(image, fx, fy, cx, cy, w_x, w_y, nside):
    """
    Project an anisotropic equidistant fisheye image onto a NESTED HEALPix grid.
    
    Args:
        image: Fisheye image tensor with shape (C, H, W).
        fx: Horizontal focal length.
        fy: Vertical focal length.
        cx: Principal-point x coordinate.
        cy: Principal-point y coordinate.
        w_x: Horizontal FoV scale.
        w_y: Vertical FoV scale.
        nside: HEALPix NSIDE value.
    
    Returns:
        A tuple containing the (C, Npix) HEALPix image and the (Npix,) validity mask.
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required for HEALPix precomputation. Install with: pip install healpy")
    
    C, H, W = image.shape
    npix = 12 * nside * nside
    device = image.device
    image_np = image.cpu().numpy()
    
    pix_indices = np.arange(npix)
    
    # HEALPix pixel -> spherical coordinates (theta, phi)
    theta, phi = hp.pix2ang(nside, pix_indices, nest=True)
    
    # Convert to 3D direction vectors (X, Y, Z).
    # In healpy, theta is colatitude in [0, pi] and phi is azimuth in [0, 2pi).
    # Convert to camera coordinates: +Z forward, +X right, +Y down.
    # Match the CUDA renderer convention p3d2lonlat: lat=atan2(Y, rho).
    # Map the HEALPix north pole (theta=0) to Y>0 using Y=cos(theta).
    X = np.sin(theta) * np.sin(phi - np.pi)  # Shift azimuth by pi so phi=pi points forward.
    Y = np.cos(theta)  # Y=cos(theta): north pole -> +1, south pole -> -1.
    Z = np.sin(theta) * np.cos(phi - np.pi)  # +Z points forward.
    
    # Initialize outputs.
    healpix_map = np.zeros((C, npix), dtype=np.float32)
    
    # Renormalize the unit directions for numerical stability.
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X = X / norm
    Y = Y / norm
    Z = Z / norm
    
    # Project 3D directions to fisheye pixel coordinates.
    # Use the inverse AnisotropicFOVCameraModel mapping.
    r_xy = np.sqrt(X**2 + Y**2)
    r_theta = np.arccos(np.clip(Z, -1.0, 1.0))  # Angle from the optical axis in [0, pi].
    
    # Handle r_xy=0 on the optical axis.
    on_axis = r_xy < 1e-10
    
    # Direction in the XY plane.
    dir_x = np.zeros_like(r_xy)
    dir_y = np.zeros_like(r_xy)
    dir_x[~on_axis] = X[~on_axis] / r_xy[~on_axis]
    dir_y[~on_axis] = Y[~on_axis] / r_xy[~on_axis]
    
    # Compute the azimuth in the normalized plane.
    # tan(phi) = (w_x * dir_y) / (w_y * dir_x)
    phi_angle = np.arctan2(w_x * dir_y, w_y * dir_x)
    
    cos_phi = np.cos(phi_angle)
    sin_phi = np.sin(phi_angle)
    
    # Compute the effective FoV scale.
    w_eff = np.sqrt((w_x * cos_phi)**2 + (w_y * sin_phi)**2)
    
    # Handle points on the optical axis.
    w_eff[on_axis] = 1.0  # Avoid division by zero.
    
    # Compute r_m = r_theta / w_eff.
    r_m = np.zeros_like(r_theta)
    valid_w = w_eff > 1e-10
    r_m[valid_w] = r_theta[valid_w] / w_eff[valid_w]
    
    # Compute normalized coordinates (m_x, m_y).
    m_x = r_m * cos_phi
    m_y = r_m * sin_phi
    
    # Handle points on the optical axis.
    m_x[on_axis] = 0.0
    m_y[on_axis] = 0.0
    
    # Convert to pixel coordinates.
    u = m_x * fx + cx
    v = m_y * fy + cy
    
    # Enforce r_theta <= w_eff*pi/2, as determined by w_x and w_y.
    # For w_x=1.222 (220-degree FoV), r_theta may reach 110 degrees.
    # Use max(w_x, w_y) as a conservative bound for anisotropic FoVs.
    max_w = max(w_x, w_y)
    in_angle = r_theta <= (max_w * np.pi / 2 + 1e-6)
    
    # Test whether samples lie inside the image.
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    # Combine validity conditions.
    final_valid = in_angle & in_bounds
    
    # Update the validity mask.
    valid_mask_updated = np.zeros(npix, dtype=bool)
    valid_mask_updated[final_valid] = True
    
    # Sample colors with nearest-neighbor lookup.
    final_valid_indices = np.where(final_valid)[0]
    u_final = np.round(u[final_valid_indices]).astype(np.int32)
    v_final = np.round(v[final_valid_indices]).astype(np.int32)
    
    # Check image bounds.
    u_final = np.clip(u_final, 0, W - 1)
    v_final = np.clip(v_final, 0, H - 1)
    
    for ch in range(C):
        healpix_map[ch, final_valid_indices] = image_np[ch, v_final, u_final]
    
    return torch.from_numpy(healpix_map).to(device), torch.from_numpy(valid_mask_updated).to(device)


def generate_fisheye_healpix_mask(fx, fy, cx, cy, w_x, w_y, width, height, nside):
    """
    Build a HEALPix visibility mask for an anisotropic fisheye camera.
    
    Returns a boolean tensor of shape (Npix,), true inside the valid fisheye region.
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required for HEALPix precomputation. Install with: pip install healpy")
    
    npix = 12 * nside * nside
    pix_indices = np.arange(npix)
    
    # HEALPix pixel -> spherical coordinates (theta, phi)
    theta, phi = hp.pix2ang(nside, pix_indices, nest=True)
    
    # Convert to 3D directions in camera coordinates.
    # Match the fisheye_to_healpix_gt convention.
    # Use Y=cos(theta), consistent with the CUDA renderer.
    X = np.sin(theta) * np.sin(phi - np.pi)
    Y = np.cos(theta)
    Z = np.sin(theta) * np.cos(phi - np.pi)
    
    # Normalize direction vectors.
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    X = X / norm
    Y = Y / norm
    Z = Z / norm
    
    r_xy = np.sqrt(X**2 + Y**2)
    r_theta = np.arccos(np.clip(Z, -1.0, 1.0))  # Angle from the optical axis in [0, pi].
    
    on_axis = r_xy < 1e-10
    
    dir_x = np.zeros_like(r_xy)
    dir_y = np.zeros_like(r_xy)
    dir_x[~on_axis] = X[~on_axis] / r_xy[~on_axis]
    dir_y[~on_axis] = Y[~on_axis] / r_xy[~on_axis]
    
    phi_angle = np.arctan2(w_x * dir_y, w_y * dir_x)
    cos_phi = np.cos(phi_angle)
    sin_phi = np.sin(phi_angle)
    
    w_eff = np.sqrt((w_x * cos_phi)**2 + (w_y * sin_phi)**2)
    w_eff[on_axis] = 1.0
    
    r_m = np.zeros_like(r_theta)
    valid_w = w_eff > 1e-10
    r_m[valid_w] = r_theta[valid_w] / w_eff[valid_w]
    
    m_x = r_m * cos_phi
    m_y = r_m * sin_phi
    m_x[on_axis] = 0.0
    m_y[on_axis] = 0.0
    
    u = m_x * fx + cx
    v = m_y * fy + cy
    
    # Enforce r_theta <= w_eff*pi/2, as determined by w_x and w_y.
    # For w_x=1.222 (220-degree FoV), r_theta may reach 110 degrees.
    # Use max(w_x, w_y) as a conservative bound for anisotropic FoVs.
    max_w = max(w_x, w_y)
    in_angle = r_theta <= (max_w * np.pi / 2 + 1e-6)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    # Combine validity conditions.
    final_valid = in_angle & in_bounds
    
    mask = final_valid.astype(bool)
    
    return torch.from_numpy(mask)


def equirectangular_to_fisheye(equi_image, fx, fy, cx, cy, w_x, w_y, width, height):
    """
    Project an equirectangular panorama to an anisotropic equidistant fisheye image.
    
    The FoV scales use pi radians as the unit, so w_x=1 represents a 180-degree
    horizontal FoV.
    
    Returns:
        Fisheye image tensor with shape (C, height, width).
    """
    C, H_equi, W_equi = equi_image.shape
    device = equi_image.device
    equi_np = equi_image.cpu().numpy()
    
    # Build the fisheye pixel grid.
    u_coords = np.arange(width)
    v_coords = np.arange(height)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    u_flat = u_grid.flatten().astype(np.float64)
    v_flat = v_grid.flatten().astype(np.float64)
    
    # Define the valid fisheye region with an elliptical boundary.
    # At r_m=a/fx, r_theta=w_x*pi/2=FoV/2.
    # Therefore a=fx*pi/2, without division by w_x.
    a = fx * np.pi / 2
    b = fy * np.pi / 2
    in_ellipse = ((u_flat - cx) / a)**2 + ((v_flat - cy) / b)**2 <= 1.0
    
    # Initialize outputs.
    fisheye_image = np.zeros((C, height * width), dtype=np.float32)
    
    valid_indices = np.where(in_ellipse)[0]
    if len(valid_indices) == 0:
        return torch.from_numpy(fisheye_image.reshape(C, height, width)).to(device)
    
    u_valid = u_flat[valid_indices]
    v_valid = v_flat[valid_indices]
     
    # fisheye -> normalized plane
    m_x = (u_valid - cx) / fx
    m_y = (v_valid - cy) / fy
    
    r_m = np.sqrt(m_x**2 + m_y**2)
    phi_angle = np.arctan2(m_y, m_x)
    
    cos_phi = np.cos(phi_angle)
    sin_phi = np.sin(phi_angle)
    w_eff = np.sqrt((w_x * cos_phi)**2 + (w_y * sin_phi)**2)
    r_theta = r_m * w_eff
    
    sin_theta = np.sin(r_theta)
    cos_theta = np.cos(r_theta)
    
    on_axis = r_m < 1e-10
    X = np.zeros_like(r_m)
    Y = np.zeros_like(r_m)
    Z = np.zeros_like(r_m)
    
    X[~on_axis] = sin_theta[~on_axis] * cos_phi[~on_axis]
    Y[~on_axis] = sin_theta[~on_axis] * sin_phi[~on_axis]
    Z[~on_axis] = cos_theta[~on_axis]
    Z[on_axis] = 1.0
    
    # Convert to equirectangular coordinates (longitude, latitude).
    # Camera coordinates: +Z forward, +X right, +Y down.
    # equirectangular: lon [-pi, pi], lat [-pi/2, pi/2]
    # Match CUDA p3d2lonlat: lat=atan2(Y, rho).
    # With +Y down, Y>0 maps to positive latitude under the renderer convention.
    
    lon = np.arctan2(X, Z)  # [-pi, pi]
    lat = np.arcsin(np.clip(Y, -1.0, 1.0))  # [-pi/2, pi/2], with Y>0 mapping to positive latitude.
    
    # Convert to equirectangular pixel coordinates.
    # Match CUDA lonlat2Equipixel_index: v=0.5+lat/pi.
    u_equi = (lon / np.pi + 1.0) * 0.5 * W_equi
    v_equi = (0.5 + lat / np.pi) * H_equi
    
    # Use nearest-neighbor sampling.
    u_equi_int = np.round(u_equi).astype(np.int32)
    v_equi_int = np.round(v_equi).astype(np.int32)
    
    # Handle horizontal wraparound at the panorama boundary.
    u_equi_int = u_equi_int % W_equi
    v_equi_int = np.clip(v_equi_int, 0, H_equi - 1)
    
    for ch in range(C):
        fisheye_image[ch, valid_indices] = equi_np[ch, v_equi_int, u_equi_int]
    
    return torch.from_numpy(fisheye_image.reshape(C, height, width)).to(device)


def healpix_to_fisheye(healpix_map, fx, fy, cx, cy, w_x, w_y, width, height, use_bilinear=True):
    """
    Synthesize an anisotropic equidistant fisheye image from a HEALPix map.
    
    The FoV scales use pi radians as the unit, so w_x=1 represents a 180-degree
    horizontal FoV. HEALPix interpolation is used when use_bilinear is true.
    
    Returns:
        Fisheye image tensor with shape (C, height, width).
    """
    if not HEALPY_AVAILABLE:
        raise RuntimeError("healpy is required. Install with: pip install healpy")
    
    C, npix = healpix_map.shape
    nside = int(np.sqrt(npix / 12))
    device = healpix_map.device
    healpix_np = healpix_map.cpu().numpy()
    
    # Build the fisheye pixel grid.
    u_coords = np.arange(width)
    v_coords = np.arange(height)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    u_flat = u_grid.flatten().astype(np.float64)
    v_flat = v_grid.flatten().astype(np.float64)
    
    # Define the valid fisheye region with an elliptical boundary.
    # At r_m=a/fx, r_theta=w_x*pi/2=FoV/2.
    # Therefore a=fx*pi/2, without division by w_x.
    a = fx * np.pi / 2
    b = fy * np.pi / 2
    in_ellipse = ((u_flat - cx) / a)**2 + ((v_flat - cy) / b)**2 <= 1.0
    
    # Initialize outputs.
    fisheye_image = np.zeros((C, height * width), dtype=np.float32)
    
    valid_indices = np.where(in_ellipse)[0]
    if len(valid_indices) == 0:
        return torch.from_numpy(fisheye_image.reshape(C, height, width)).to(device)
    
    u_valid = u_flat[valid_indices]
    v_valid = v_flat[valid_indices]
    
    # fisheye -> normalized plane
    m_x = (u_valid - cx) / fx
    m_y = (v_valid - cy) / fy
    
    r_m = np.sqrt(m_x**2 + m_y**2)
    phi_angle = np.arctan2(m_y, m_x)
    
    cos_phi = np.cos(phi_angle)
    sin_phi = np.sin(phi_angle)
    w_eff = np.sqrt((w_x * cos_phi)**2 + (w_y * sin_phi)**2)
    r_theta = r_m * w_eff
    
    sin_theta = np.sin(r_theta)
    cos_theta = np.cos(r_theta)
    
    on_axis = r_m < 1e-10
    X = np.zeros_like(r_m)
    Y = np.zeros_like(r_m)
    Z = np.zeros_like(r_m)
    
    X[~on_axis] = sin_theta[~on_axis] * cos_phi[~on_axis]
    Y[~on_axis] = sin_theta[~on_axis] * sin_phi[~on_axis]
    Z[~on_axis] = cos_theta[~on_axis]
    Z[on_axis] = 1.0
    
    # Convert to the HEALPix coordinate system.
    # Camera coordinates: +Z forward, +X right, +Y down.
    # HEALPix uses colatitude theta in [0, pi] and azimuth phi in [0, 2pi).
    # Follow the inverse transform used by healpix_to_perspective:
    # X=sin(theta)sin(phi-pi), Y=cos(theta), Z=sin(theta)cos(phi-pi).
    # Thus theta=arccos(Y) and phi=atan2(X,Z)+pi.
    
    # Colatitude theta is measured from the +Y axis.
    theta = np.arccos(np.clip(Y, -1.0, 1.0))  # [0, π]
    
    # Azimuth phi is measured in the XZ plane and shifted by pi.
    phi = np.arctan2(X, Z) + np.pi  # [0, 2π)
    
    # Wrap phi to [0, 2pi).
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    phi = np.where(phi >= 2 * np.pi, phi - 2 * np.pi, phi)
    
    if use_bilinear:
        # Use healpy interpolation to reduce aliasing.
        for ch in range(C):
            # Convert NESTED order to RING order for interpolation.
            healpix_ring = hp.reorder(healpix_np[ch], n2r=True)
            # Interpolate the HEALPix samples.
            sampled = hp.get_interp_val(healpix_ring, theta, phi)
            fisheye_image[ch, valid_indices] = sampled
    else:
        # Nearest-neighbor sampling is faster but introduces aliasing.
        pix_indices = hp.ang2pix(nside, theta, phi, nest=True)
        for ch in range(C):
            fisheye_image[ch, valid_indices] = healpix_np[ch, pix_indices]
    
    return torch.from_numpy(fisheye_image.reshape(C, height, width)).to(device)


class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        
        self._original_hp_cache = {}

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]
            # If no depth_params, use depth map directly (already inverse depth)

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_healpix_gt(self, healpix_scale):
        if healpix_scale not in self._original_hp_cache:
            nside = 2 ** healpix_scale
            self._original_hp_cache[healpix_scale] = equirectangular_to_healpix_gt(
                self.original_image, nside
            )
        return self._original_hp_cache[healpix_scale]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

