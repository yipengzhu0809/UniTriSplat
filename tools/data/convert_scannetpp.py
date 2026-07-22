#!/usr/bin/env python3
"""
Convert ScanNet++ DSLR fisheye images to the equidistant model.

The conversion preserves the source dimensions and matches incidence angles
at the image boundary through inverse mapping and bilinear sampling.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import struct
from pathlib import Path
from scipy.optimize import brentq


def read_colmap_cameras_txt(path):
    """
    Read cameras from a COLMAP cameras.txt file.
    """
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_colmap_images_txt(path):
    """
    Read registered images from a COLMAP images.txt file.
    """
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        
        images[image_id] = {
            'qvec': [qw, qx, qy, qz],
            'tvec': [tx, ty, tz],
            'camera_id': camera_id,
            'name': name
        }
        
        i += 2  # Skip the POINTS2D line.
    
    return images


def read_colmap_cameras_bin(path):
    """
    Read cameras from a COLMAP cameras.bin file.
    """
    cameras = {}
    MODEL_PARAMS = {
        0: ('SIMPLE_PINHOLE', 3),
        1: ('PINHOLE', 4),
        2: ('SIMPLE_RADIAL', 4),
        3: ('RADIAL', 5),
        4: ('OPENCV', 8),
        5: ('OPENCV_FISHEYE', 8),
        6: ('FULL_OPENCV', 12),
        8: ('SIMPLE_RADIAL_FISHEYE', 4),
        9: ('RADIAL_FISHEYE', 5),
    }
    
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<I', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            
            if model_id not in MODEL_PARAMS:
                raise ValueError(f"Unsupported camera model ID: {model_id}")
            
            model_name, num_params = MODEL_PARAMS[model_id]
            
            if model_id != 5:
                raise ValueError(f"Only OPENCV_FISHEYE (model_id=5) is supported, got {model_name}")
            
            params = struct.unpack('<' + 'd' * num_params, f.read(8 * num_params))
            
            cameras[camera_id] = {
                'model': 'OPENCV_FISHEYE',
                'width': width,
                'height': height,
                'params': list(params)
            }
    return cameras


def read_colmap_images_bin(path):
    """
    Read registered images from a COLMAP images.bin file.
    """
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
            tx, ty, tz = struct.unpack('<ddd', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]
            
            name_chars = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_chars.append(c.decode('utf-8'))
            name = ''.join(name_chars)
            
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2D * 24)
            
            images[image_id] = {
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz],
                'camera_id': camera_id,
                'name': name
            }
    
    return images


def qvec2rotmat(qvec):
    """
    Convert a quaternion to a rotation matrix.
    """
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def get_c2w_matrix(qvec, tvec):
    """
    Convert COLMAP world-to-camera qvec and tvec to a camera-to-world matrix.
    """
    R = qvec2rotmat(qvec)
    t = np.array(tvec).reshape(3, 1)
    R_c2w = R.T
    t_c2w = -R.T @ t
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w.flatten()
    return c2w


def opencv_fisheye_theta_d(theta, k1, k2, k3, k4):
    """
    Compute the distorted OPENCV_FISHEYE angle theta_d.
    """
    theta2 = theta ** 2
    theta4 = theta2 ** 2
    theta6 = theta4 * theta2
    theta8 = theta4 ** 2
    return theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)


def opencv_fisheye_inverse_theta(theta_d, k1, k2, k3, k4, max_iter=20):
    """
    Recover the incidence angle theta from theta_d with Brent root finding.
    """
    if theta_d < 1e-10:
        return 0.0
    
    # theta_d = theta * (1 + k*theta^2 + ...)
    # Search theta within [0, 1.5*theta_d].
    def f(theta):
        return opencv_fisheye_theta_d(theta, k1, k2, k3, k4) - theta_d
    
    # Find a valid root-search interval.
    theta_max = min(theta_d * 1.5, np.pi / 2)
    try:
        theta = brentq(f, 0, theta_max)
    except ValueError:
        # Fall back to an approximate value when no root is bracketed.
        theta = theta_d
    
    return theta


def opencv_fisheye_project(X, Y, Z, fx, fy, cx, cy, k1, k2, k3, k4):
    """
    Project camera-space 3D points with the OPENCV_FISHEYE model.
    """
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r, Z)
    
    theta2 = theta ** 2
    theta4 = theta2 ** 2
    theta6 = theta4 * theta2
    theta8 = theta4 ** 2
    theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
    
    r_safe = np.where(r > 1e-10, r, 1.0)
    scale = np.where(r > 1e-10, theta_d / r_safe, 1.0)
    
    x_prime = scale * X
    y_prime = scale * Y
    
    u = fx * x_prime + cx
    v = fy * y_prime + cy
    
    return u, v


def equidistant_unproject(u, v, fx, fy, cx, cy):
    """Unproject conventional equidistant pixels to unit camera rays."""
    m_x = (u - cx) / fx
    m_y = (v - cy) / fy
    
    theta = np.sqrt(m_x**2 + m_y**2)
    phi = np.arctan2(m_y, m_x)
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    on_axis = theta < 1e-10
    
    X = np.where(on_axis, 0.0, sin_theta * np.cos(phi))
    Y = np.where(on_axis, 0.0, sin_theta * np.sin(phi))
    Z = np.where(on_axis, 1.0, cos_theta)
    
    valid = theta <= np.pi / 2
    
    return X, Y, Z, valid


def bilinear_sample(image, u, v):
    """
    Sample an image at array-valued pixel coordinates with bilinear interpolation.
    """
    H, W = image.shape[:2]
    
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    
    wu = u - u0
    wv = v - v0
    
    if image.ndim == 2:
        val = (1 - wu) * (1 - wv) * image[v0, u0] + \
              wu * (1 - wv) * image[v0, u1] + \
              (1 - wu) * wv * image[v1, u0] + \
              wu * wv * image[v1, u1]
    else:
        wu = wu[..., np.newaxis]
        wv = wv[..., np.newaxis]
        val = (1 - wu) * (1 - wv) * image[v0, u0] + \
              wu * (1 - wv) * image[v0, u1] + \
              (1 - wu) * wv * image[v1, u0] + \
              wu * wv * image[v1, u1]
    
    return val


def compute_equidistant_params(src_params, src_width, src_height):
    """
    Match equidistant intrinsics to source incidence angles at the image boundary.
    """
    fx_src, fy_src, cx_src, cy_src, k1, k2, k3, k4 = src_params
    
    # Compute source incidence angles at the image boundary.
    # Evaluate the centers of the right and bottom boundaries.
    r_edge_x = (src_width - cx_src)  # Pixel distance to the right boundary.
    r_edge_y = (src_height - cy_src)  # Pixel distance to the bottom boundary.
    
    # Normalize image coordinates.
    theta_d_x = r_edge_x / fx_src
    theta_d_y = r_edge_y / fy_src
    
    # Recover the physical incidence angles.
    theta_x = opencv_fisheye_inverse_theta(theta_d_x, k1, k2, k3, k4)
    theta_y = opencv_fisheye_inverse_theta(theta_d_y, k1, k2, k3, k4)
    
    print(f"  Source camera boundary:")
    print(f"    Horizontal: theta_d={np.degrees(theta_d_x):.2f}°, theta={np.degrees(theta_x):.2f}°")
    print(f"    Vertical: theta_d={np.degrees(theta_d_y):.2f}°, theta={np.degrees(theta_y):.2f}°")
    
    # Preserve the calibrated principal point and match the incidence angle
    # at each image boundary with the conventional equidistant model.
    fx_dst = r_edge_x / theta_x
    fy_dst = r_edge_y / theta_y
    cx_dst = cx_src
    cy_dst = cy_src
    fov_x = 2 * theta_x
    fov_y = 2 * theta_y
    
    print(f"  Target equidistant parameters:")
    print(f"    fx={fx_dst:.2f}, fy={fy_dst:.2f}")
    print(f"    cx={cx_dst:.2f}, cy={cy_dst:.2f}")
    print(f"    FOV: {np.degrees(fov_x):.1f}° x {np.degrees(fov_y):.1f}°")
    
    return {
        'fx': fx_dst,
        'fy': fy_dst,
        'cx': cx_dst,
        'cy': cy_dst,
        'w_x': 1.0,
        'w_y': 1.0,
        'width': src_width,
        'height': src_height,
        'fov_x': fov_x,
        'fov_y': fov_y
    }


def convert_image(src_image, src_params, dst_params):
    """
    Convert an OPENCV_FISHEYE image to the equidistant model.
    """
    dst_height = dst_params['height']
    dst_width = dst_params['width']
    
    # Build the target pixel grid.
    u_dst = np.arange(dst_width).astype(np.float64)
    v_dst = np.arange(dst_height).astype(np.float64)
    u_grid, v_grid = np.meshgrid(u_dst, v_dst)
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    
    # Target pixels -> 3D rays.
    X, Y, Z, valid = equidistant_unproject(
        u_flat, v_flat,
        dst_params['fx'], dst_params['fy'],
        dst_params['cx'], dst_params['cy']
    )
    
    # 3D rays -> source pixels.
    u_src, v_src = opencv_fisheye_project(
        X, Y, Z,
        src_params['fx'], src_params['fy'],
        src_params['cx'], src_params['cy'],
        src_params['k1'], src_params['k2'],
        src_params['k3'], src_params['k4']
    )
    
    # Test whether samples lie inside the source image.
    H_src, W_src = src_image.shape[:2]
    in_bounds = (u_src >= 0) & (u_src < W_src) & (v_src >= 0) & (v_src < H_src)
    valid = valid & in_bounds
    
    # Initialize the output.
    C = src_image.shape[2] if src_image.ndim == 3 else 1
    if C == 1:
        dst_image = np.zeros((dst_height * dst_width,), dtype=np.uint8)
    else:
        dst_image = np.zeros((dst_height * dst_width, C), dtype=np.uint8)
    
    # Sample valid pixels.
    valid_indices = np.where(valid)[0]
    if len(valid_indices) > 0:
        sampled = bilinear_sample(src_image, u_src[valid_indices], v_src[valid_indices])
        dst_image[valid_indices] = np.clip(sampled, 0, 255).astype(np.uint8)
    
    if C == 1:
        return dst_image.reshape(dst_height, dst_width)
    else:
        return dst_image.reshape(dst_height, dst_width, C)


def find_image_path(img_name, images_dir):
    """
    Locate an image using the supported ScanNet++ path conventions.
    """
    path = os.path.join(images_dir, img_name)
    if os.path.exists(path):
        return path
    
    name_no_prefix = img_name
    for prefix in ['cam1/', 'cam2/', 'cam1\\', 'cam2\\']:
        if img_name.startswith(prefix):
            name_no_prefix = img_name[len(prefix):]
            break
    path = os.path.join(images_dir, name_no_prefix)
    if os.path.exists(path):
        return path
    
    import re
    name_no_suffix = re.sub(r'_fisheye[12](\.[^.]+)$', r'\1', name_no_prefix)
    path = os.path.join(images_dir, name_no_suffix)
    if os.path.exists(path):
        return path
    
    basename = os.path.basename(name_no_suffix)
    path = os.path.join(images_dir, basename)
    if os.path.exists(path):
        return path
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert ScanNet++ DSLR fisheye images to the equidistant model")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing colmap/ and resized_images/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--colmap_subdir", type=str, default="colmap",
                        help="COLMAP subdirectory (default: colmap)")
    parser.add_argument("--images_subdir", type=str, default="resized_images",
                        help="Image subdirectory (default: resized_images)")
    parser.add_argument("--test_every", type=int, default=8,
                        help="Use every Nth image for testing (default: 8)")
    
    args = parser.parse_args()
    
    # Read COLMAP data.
    colmap_dir = os.path.join(args.input_dir, args.colmap_subdir)
    
    cameras_txt = os.path.join(colmap_dir, 'cameras.txt')
    cameras_bin = os.path.join(colmap_dir, 'cameras.bin')
    images_txt = os.path.join(colmap_dir, 'images.txt')
    images_bin = os.path.join(colmap_dir, 'images.bin')
    
    if os.path.exists(cameras_txt):
        cameras = read_colmap_cameras_txt(cameras_txt)
        print(f"Read cameras.txt: {len(cameras)} cameras")
    elif os.path.exists(cameras_bin):
        cameras = read_colmap_cameras_bin(cameras_bin)
        print(f"Read cameras.bin: {len(cameras)} cameras")
    else:
        raise FileNotFoundError(f"Could not find cameras.txt or cameras.bin")
    
    if os.path.exists(images_txt):
        images = read_colmap_images_txt(images_txt)
        print(f"Read images.txt: {len(images)} images")
    elif os.path.exists(images_bin):
        images = read_colmap_images_bin(images_bin)
        print(f"Read images.bin: {len(images)} images")
    else:
        raise FileNotFoundError(f"Could not find images.txt or images.bin")
    
    # Read camera parameters.
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    src_width = cam['width']
    src_height = cam['height']
    params = cam['params']
    
    print(f"\nSource camera ({cam['model']}):")
    print(f"  Resolution: {src_width} x {src_height}")
    print(f"  fx={params[0]:.2f}, fy={params[1]:.2f}")
    print(f"  cx={params[2]:.2f}, cy={params[3]:.2f}")
    print(f"  k1={params[4]:.6f}, k2={params[5]:.6f}, k3={params[6]:.6f}, k4={params[7]:.6f}")
    
    src_params = {
        'fx': params[0], 'fy': params[1],
        'cx': params[2], 'cy': params[3],
        'k1': params[4], 'k2': params[5],
        'k3': params[6], 'k4': params[7]
    }
    
    # Compute matched equidistant parameters.
    print(f"\nComputing equidistant parameters...")
    dst_params = compute_equidistant_params(params, src_width, src_height)
    
    # Create output directories.
    output_images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Split training and test sets.
    image_ids = sorted(images.keys())
    test_ids = set(image_ids[::args.test_every])
    train_ids = [i for i in image_ids if i not in test_ids]
    test_ids = list(test_ids)
    
    print(f"\nDataset split: {len(train_ids)} train / {len(test_ids)} test")
    
    # Process images.
    images_dir = os.path.join(args.input_dir, args.images_subdir)
    train_frames = []
    test_frames = []
    
    for image_id in tqdm(image_ids, desc="Converting images"):
        img_info = images[image_id]
        img_name = img_info['name']
        img_path = find_image_path(img_name, images_dir)
        
        if img_path is None:
            print(f"Warning: image not found: {img_name}")
            continue
        
        # Read and convert the image.
        src_image = np.array(Image.open(img_path))
        dst_image = convert_image(src_image, src_params, dst_params)
        
        # Save the converted image.
        img_name_normalized = img_name.replace('/', '_').replace('\\', '_')
        base_name = os.path.splitext(img_name_normalized)[0]
        out_name = f"{base_name}.jpg"
        out_path = os.path.join(output_images_dir, out_name)
        Image.fromarray(dst_image).save(out_path, quality=95)
        
        # Camera-to-world matrix.
        c2w = get_c2w_matrix(img_info['qvec'], img_info['tvec'])
        
        frame = {
            "rgb_file": out_name,
            "transform_matrix": c2w.tolist()
        }
        
        if image_id in test_ids:
            test_frames.append(frame)
        else:
            train_frames.append(frame)
    
    # Save transforms.
    camera_angle_x = dst_params['fov_x']
    
    output_common = {
        "camera_angle_x": camera_angle_x,
        "fisheye_fx": dst_params['fx'],
        "fisheye_fy": dst_params['fy'],
        "fisheye_cx": dst_params['cx'],
        "fisheye_cy": dst_params['cy'],
        "fisheye_w_x": dst_params['w_x'],
        "fisheye_w_y": dst_params['w_y'],
        "fisheye_width": dst_params['width'],
        "fisheye_height": dst_params['height'],
        "fisheye_fov_x": np.degrees(dst_params['fov_x']),
        "fisheye_fov_y": np.degrees(dst_params['fov_y']),
    }
    
    output_train = {**output_common, "frames": train_frames}
    output_test = {**output_common, "frames": test_frames}
    
    train_path = os.path.join(args.output_dir, "transforms_train.json")
    test_path = os.path.join(args.output_dir, "transforms_test.json")
    
    with open(train_path, 'w') as f:
        json.dump(output_train, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(output_test, f, indent=2)
    
    print(f"\nDone.")
    print(f"  Training set: {len(train_frames)} images -> {train_path}")
    print(f"  Test set: {len(test_frames)} images -> {test_path}")
    print(f"  Image directory: {output_images_dir}")
    print(f"\nExample training command:")
    print(f"  python train.py -s {args.output_dir} --camera_model 1")


if __name__ == "__main__":
    main()
