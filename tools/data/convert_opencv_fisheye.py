#!/usr/bin/env python3
"""
Convert COLMAP OPENCV_FISHEYE images to the anisotropic equidistant model.

The conversion uses inverse mapping: unproject each target pixel to a 3D ray,
project the ray with the source OPENCV_FISHEYE model, and bilinearly sample
the source image.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import struct
from pathlib import Path


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
    # COLMAP camera model IDs
    # 0: SIMPLE_PINHOLE, 1: PINHOLE, 2: SIMPLE_RADIAL, 3: RADIAL
    # 4: OPENCV, 5: OPENCV_FISHEYE, 6: FULL_OPENCV, 7: FOV
    # 8: SIMPLE_RADIAL_FISHEYE, 9: RADIAL_FISHEYE, 10: THIN_PRISM_FISHEYE
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
            
            # Only OPENCV_FISHEYE (model_id=5) is supported.
            if model_id != 5:
                raise ValueError(f"Only OPENCV_FISHEYE (model_id=5) is supported, got {model_name} (model_id={model_id})")
            
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
            
            # Read the null-terminated file name.
            name_chars = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_chars.append(c.decode('utf-8'))
            name = ''.join(name_chars)
            
            # Skip POINTS2D records.
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2D * 24)  # Each point stores x(8), y(8), and point3D_id(8).
            
            images[image_id] = {
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz],
                'camera_id': camera_id,
                'name': name
            }
    
    return images


def qvec2rotmat(qvec):
    """
    Convert a COLMAP quaternion (qw, qx, qy, qz) to a rotation matrix.
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
    
    # COLMAP stores world-to-camera poses; invert them here.
    R_c2w = R.T
    t_c2w = -R.T @ t
    
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w.flatten()
    
    return c2w


def opencv_fisheye_project(X, Y, Z, fx, fy, cx, cy, k1, k2, k3, k4):
    """
    Project camera-space 3D points with the COLMAP OPENCV_FISHEYE model.
    
    Returns:
        Pixel coordinates u and v.
    """
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r, Z)  # Incidence angle.
    
    # Apply fisheye distortion.
    theta2 = theta ** 2
    theta4 = theta2 ** 2
    theta6 = theta4 * theta2
    theta8 = theta4 ** 2
    theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
    
    # Project to pixels.
    # Avoid division by zero.
    r_safe = np.where(r > 1e-10, r, 1.0)
    scale = np.where(r > 1e-10, theta_d / r_safe, 1.0)
    
    x_prime = scale * X
    y_prime = scale * Y
    
    u = fx * x_prime + cx
    v = fy * y_prime + cy
    
    return u, v


def equidistant_unproject(u, v, fx, fy, cx, cy, w_x, w_y):
    """
    Unproject anisotropic equidistant pixels to unit camera rays.
    
    The FoV scales use pi radians as the unit, so 1.0 represents 180 degrees.
    
    Returns:
        Unit ray components X, Y, Z and a validity mask.
    """
    m_x = (u - cx) / fx
    m_y = (v - cy) / fy
    
    r_m = np.sqrt(m_x**2 + m_y**2)
    phi = np.arctan2(m_y, m_x)
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    w_eff = np.sqrt((w_x * cos_phi)**2 + (w_y * sin_phi)**2)
    
    # Incidence angle.
    theta = r_m * w_eff
    
    # Valid region: theta <= max_w*pi/2.
    max_w = max(w_x, w_y)
    valid = theta <= max_w * np.pi / 2
    
    # Compute 3D ray directions.
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Handle r_m=0.
    on_axis = r_m < 1e-10
    
    X = np.where(on_axis, 0.0, sin_theta * cos_phi)
    Y = np.where(on_axis, 0.0, sin_theta * sin_phi)
    Z = np.where(on_axis, 1.0, cos_theta)
    
    return X, Y, Z, valid


def bilinear_sample(image, u, v):
    """
    Sample an image at array-valued pixel coordinates with bilinear interpolation.
    """
    H, W = image.shape[:2]
    C = image.shape[2] if image.ndim == 3 else 1
    
    # Check image bounds.
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


def convert_fisheye_image(src_image, src_params, dst_params, dst_width, dst_height):
    """
    Convert an OPENCV_FISHEYE image to an anisotropic equidistant image.
    
    Source and destination intrinsics are supplied as dictionaries. The output
    shape is determined by dst_width and dst_height.
    """
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
        dst_params['cx'], dst_params['cy'],
        dst_params['w_x'], dst_params['w_y']
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


def main():
    parser = argparse.ArgumentParser(description="Convert OPENCV_FISHEYE images to the equidistant model")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing colmap/model and images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--colmap_subdir", type=str, default="colmap/model",
                        help="COLMAP model subdirectory (default: colmap/model)")
    parser.add_argument("--images_subdir", type=str, default="images",
                        help="Image subdirectory (default: images)")
    
    # Target fisheye parameters.
    parser.add_argument("--dst_width", type=int, default=1000,
                        help="Target image width")
    parser.add_argument("--dst_height", type=int, default=1000,
                        help="Target image height")
    parser.add_argument("--dst_fov", type=float, default=180.0,
                        help="Target FoV in degrees, e.g. 180 or 220")
    parser.add_argument("--dst_fx", type=float, default=-1,
                        help="Target focal length fx (-1 computes the tangent fit automatically)")
    parser.add_argument("--dst_fy", type=float, default=-1,
                        help="Target focal length fy (-1 computes the tangent fit automatically)")
    
    # Dataset split.
    parser.add_argument("--test_ratio", type=float, default=0.125,
                        help="Test-set ratio (default: 0.125 = 1/8)")
    parser.add_argument("--test_every", type=int, default=-1,
                        help="Use every Nth image for testing (-1 uses test_ratio)")
    
    # Camera filtering.
    parser.add_argument("--camera_prefix", type=str, default=None,
                        help="Process only images with this prefix, e.g. cam1")
    
    # Generate transforms.json without converting images.
    parser.add_argument("--no_convert", action="store_true",
                        help="Generate transforms.json only (approximation: ignore k1-k4 and set w_x=w_y=1)")
    
    args = parser.parse_args()
    
    # Compute target parameters.
    w_x = args.dst_fov / 180.0  # FOV = w_x * 180
    w_y = w_x  # Isotropic FoV.
    
    cx = args.dst_width / 2.0
    cy = args.dst_height / 2.0
    
    # Compute fx automatically so the valid circle is tangent to the boundary.
    # a = fx * pi / 2 = cx  =>  fx = 2 * cx / pi
    if args.dst_fx < 0:
        fx = args.dst_width / np.pi
    else:
        fx = args.dst_fx
    
    if args.dst_fy < 0:
        fy = args.dst_height / np.pi
    else:
        fy = args.dst_fy
    
    dst_params = {
        'fx': fx, 'fy': fy,
        'cx': cx, 'cy': cy,
        'w_x': w_x, 'w_y': w_y
    }
    
    print(f"Target fisheye parameters:")
    print(f"  Resolution: {args.dst_width} x {args.dst_height}")
    print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  FOV: {args.dst_fov}° (w_x={w_x:.3f}, w_y={w_y:.3f})")
    
    # Read COLMAP data.
    colmap_dir = os.path.join(args.input_dir, args.colmap_subdir)
    
    # Prefer text model files when available.
    cameras_txt = os.path.join(colmap_dir, 'cameras.txt')
    cameras_bin = os.path.join(colmap_dir, 'cameras.bin')
    images_txt = os.path.join(colmap_dir, 'images.txt')
    images_bin = os.path.join(colmap_dir, 'images.bin')
    
    if os.path.exists(cameras_txt):
        cameras = read_colmap_cameras_txt(cameras_txt)
        print(f"\nRead cameras.txt: {len(cameras)} cameras")
    elif os.path.exists(cameras_bin):
        cameras = read_colmap_cameras_bin(cameras_bin)
        print(f"\nRead cameras.bin: {len(cameras)} cameras")
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
    
    # Print camera parameters.
    for cam_id, cam in cameras.items():
        print(f"\nCamera {cam_id}: {cam['model']}")
        print(f"  Resolution: {cam['width']} x {cam['height']}")
        if cam['model'] == 'OPENCV_FISHEYE':
            params = cam['params']
            print(f"  fx={params[0]:.2f}, fy={params[1]:.2f}")
            print(f"  cx={params[2]:.2f}, cy={params[3]:.2f}")
            print(f"  k1={params[4]:.6f}, k2={params[5]:.6f}")
            print(f"  k3={params[6]:.6f}, k4={params[7]:.6f}")
    
    # Create output directories.
    output_images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Filter images by camera prefix.
    image_ids = sorted(images.keys())
    if args.camera_prefix is not None:
        image_ids = [i for i in image_ids if images[i]['name'].startswith(args.camera_prefix)]
        print(f"\nCamera prefix '{args.camera_prefix}': {len(image_ids)} images")
    
    # Split training and test sets.
    if args.test_every > 0:
        test_ids = set(image_ids[::args.test_every])
    else:
        test_count = max(1, int(len(image_ids) * args.test_ratio))
        step = len(image_ids) // test_count
        test_ids = set(image_ids[::step][:test_count])
    
    train_ids = [i for i in image_ids if i not in test_ids]
    test_ids = [i for i in image_ids if i in test_ids]
    
    print(f"Dataset split: {len(train_ids)} train / {len(test_ids)} test")
    
    # Initialize transform records.
    train_frames = []
    test_frames = []
    
    # Process each image.
    images_dir = os.path.join(args.input_dir, args.images_subdir)
    
    def find_image_path(img_name, images_dir):
        """
        Locate an image using the supported COLMAP path conventions.
        """
        # Strategy 1: use the path stored by COLMAP.
        path = os.path.join(images_dir, img_name)
        if os.path.exists(path):
            return path
        
        # Strategy 2: remove a cam1/ or cam2/ prefix.
        name_no_prefix = img_name
        for prefix in ['cam1/', 'cam2/', 'cam1\\', 'cam2\\']:
            if img_name.startswith(prefix):
                name_no_prefix = img_name[len(prefix):]
                break
        path = os.path.join(images_dir, name_no_prefix)
        if os.path.exists(path):
            return path
        
        # Strategy 3: remove a _fisheye1 or _fisheye2 suffix.
        import re
        name_no_suffix = re.sub(r'_fisheye[12](\.[^.]+)$', r'\1', name_no_prefix)
        path = os.path.join(images_dir, name_no_suffix)
        if os.path.exists(path):
            return path
        
        # Strategy 4: search images by the normalized base name.
        basename = os.path.basename(name_no_suffix)
        path = os.path.join(images_dir, basename)
        if os.path.exists(path):
            return path
        
        return None
    
    # Use the first camera parameters in --no_convert mode.
    first_cam = cameras[list(cameras.keys())[0]]
    src_width = first_cam['width']
    src_height = first_cam['height']
    src_cam_params = first_cam['params']
    
    for image_id in tqdm(image_ids, desc="Processing images"):
        img_info = images[image_id]
        cam_info = cameras[img_info['camera_id']]
        
        # Resolve the source image path.
        img_name = img_info['name']
        img_path = find_image_path(img_name, images_dir)
        
        if img_path is None:
            print(f"Warning: image not found: {os.path.join(images_dir, img_name)}")
            continue
        
        # Parse source camera parameters.
        params = cam_info['params']
        src_params = {
            'fx': params[0], 'fy': params[1],
            'cx': params[2], 'cy': params[3],
            'k1': params[4], 'k2': params[5],
            'k3': params[6], 'k4': params[7]
        }
        
        # Build the output file name.
        img_name_normalized = img_name.replace('/', '_').replace('\\', '_')
        base_name = os.path.splitext(img_name_normalized)[0]
        
        if args.no_convert:
            # Preserve the image through a symbolic link.
            ext = os.path.splitext(img_path)[1]
            out_name = f"{base_name}{ext}"
            out_path = os.path.join(output_images_dir, out_name)
            if not os.path.exists(out_path):
                os.symlink(os.path.abspath(img_path), out_path)
        else:
            # Read and convert the source image.
            src_image = np.array(Image.open(img_path))
            dst_image = convert_fisheye_image(
                src_image, src_params, dst_params,
                args.dst_width, args.dst_height
            )
            out_name = f"{base_name}.jpg"
            out_path = os.path.join(output_images_dir, out_name)
            Image.fromarray(dst_image).save(out_path, quality=95)
        
        # Compute the camera-to-world matrix.
        c2w = get_c2w_matrix(img_info['qvec'], img_info['tvec'])
        
        # Add the frame record.
        frame = {
            "rgb_file": out_name,
            "transform_matrix": c2w.tolist()
        }
        
        if image_id in test_ids:
            test_frames.append(frame)
        else:
            train_frames.append(frame)
    
    # Save transforms.
    if args.no_convert:
        # Approximation: ignore k1-k4 and set w_x=w_y=1.0.
        # OPENCV_FISHEYE with zero distortion is equivalent to equidistant w=1.
        # FoV follows from image size and focal length: FoV approximately width/fx radians.
        camera_angle_x = src_width / src_cam_params[0]  # Effective horizontal FoV in radians.
        output_train = {
            "camera_angle_x": camera_angle_x,
            "fisheye_fx": src_cam_params[0],
            "fisheye_fy": src_cam_params[1],
            "fisheye_cx": src_cam_params[2],
            "fisheye_cy": src_cam_params[3],
            "fisheye_w_x": 1.0,
            "fisheye_w_y": 1.0,
            "fisheye_width": src_width,
            "fisheye_height": src_height,
            "frames": train_frames
        }
        output_test = {
            "camera_angle_x": camera_angle_x,
            "fisheye_fx": src_cam_params[0],
            "fisheye_fy": src_cam_params[1],
            "fisheye_cx": src_cam_params[2],
            "fisheye_cy": src_cam_params[3],
            "fisheye_w_x": 1.0,
            "fisheye_w_y": 1.0,
            "fisheye_width": src_width,
            "fisheye_height": src_height,
            "frames": test_frames
        }
    else:
        # Use equidistant parameters.
        camera_angle_x = dst_params['w_x'] * np.pi  # FOV in radians
        output_train = {
            "camera_angle_x": camera_angle_x,
            "fisheye_fx": dst_params['fx'],
            "fisheye_fy": dst_params['fy'],
            "fisheye_cx": dst_params['cx'],
            "fisheye_cy": dst_params['cy'],
            "fisheye_w_x": dst_params['w_x'],
            "fisheye_w_y": dst_params['w_y'],
            "fisheye_width": args.dst_width,
            "fisheye_height": args.dst_height,
            "frames": train_frames
        }
        output_test = {
            "camera_angle_x": camera_angle_x,
            "fisheye_fx": dst_params['fx'],
            "fisheye_fy": dst_params['fy'],
            "fisheye_cx": dst_params['cx'],
            "fisheye_cy": dst_params['cy'],
            "fisheye_w_x": dst_params['w_x'],
            "fisheye_w_y": dst_params['w_y'],
            "fisheye_width": args.dst_width,
            "fisheye_height": args.dst_height,
            "frames": test_frames
        }
    
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
    if args.no_convert:
        actual_fov_x = src_width / src_cam_params[0] * 180 / np.pi
        actual_fov_y = src_height / src_cam_params[1] * 180 / np.pi
        print(f"  Mode: preserve original images with symbolic links")
        print(f"  Parameter mapping: OPENCV_FISHEYE -> equidistant (ignore k1-k4, w_x=w_y=1.0)")
        print(f"  Effective FoV: {actual_fov_x:.1f}° x {actual_fov_y:.1f}°")
        print(f"  k1={src_cam_params[4]:.6f}, k2={src_cam_params[5]:.6f}, k3={src_cam_params[6]:.6f}, k4={src_cam_params[7]:.6f} (ignored)")
    else:
        print(f"  Mode: convert to equidistant fisheye")


if __name__ == "__main__":
    main()
