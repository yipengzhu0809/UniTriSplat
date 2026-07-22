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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## depth_params.json is optional - if not found, use depth maps directly
    depths_params = None
    if depths != "":
        if os.path.exists(depth_params_file):
            try:
                with open(depth_params_file, "r") as f:
                    depths_params = json.load(f)
                all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
                if (all_scales > 0).sum():
                    med_scale = np.median(all_scales[all_scales > 0])
                else:
                    med_scale = 0
                for key in depths_params:
                    depths_params[key]["med_scale"] = med_scale
                print(f"Loaded depth_params.json with {len(depths_params)} entries")
            except Exception as e:
                print(f"Warning: Failed to load depth_params.json: {e}")
                depths_params = None
        else:
            print(f"Note: depth_params.json not found, using depth maps directly without scale/offset correction")

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            if "rgb_file" in frame:
                cam_name = os.path.join(path, './images/'+frame["rgb_file"])
            else:    
                cam_name = os.path.join(path, './images/'+frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            if "rgb_file" not in frame: # for the 360roam dataset, the coordinate already matches COLMAP
                c2w[:3, 1:3] *= -1
            # c2w[:3, 1:3] *= -1
            

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            rgb = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(rgb, "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}


# ===================== Cubemap Support =====================
# The following functions support reading cubemap data from the cubemap folder
# without modifying the original data reading logic

CUBEMAP_FACES = ['front', 'back', 'left', 'right', 'up', 'down']


def readCamerasFromCubemapTransforms(cubemap_path, base_path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    """
    Read cameras from a cubemap transforms JSON file.
    Similar to readCamerasFromTransforms but for cubemap faces.
    
    Args:
        cubemap_path: Path to the cubemap folder (where JSON files are)
        base_path: Base path to the dataset (where images/ folder is)
        transformsfile: Name of the transforms JSON file
        depths_folder: Depth folder path
        white_background: Whether to use white background
        is_test: Whether this is test set
        extension: Image file extension
    """
    cam_infos = []
    
    json_path = os.path.join(cubemap_path, transformsfile)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skipping...")
        return cam_infos

    with open(json_path) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        # Convert degrees to radians if needed (cubemap should be 90 degrees = pi/2)
        if fovx > 3.15:  # If greater than pi, it's in degrees
            fovx = np.radians(fovx)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # Images are in the base_path/images/ folder, not in cubemap/images/
            if "rgb_file" in frame:
                cam_name = os.path.join(base_path, './images/' + frame["rgb_file"])
            else:    
                cam_name = os.path.join(base_path, './images/' + frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # For cubemap data generated from 360roam, coordinate already matches COLMAP
            # (no need to flip Y and Z axes)
            if "rgb_file" not in frame: # for the 360roam dataset, the coordinate already matches COLMAP
                c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name  # cam_name is already the full path
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            rgb = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(rgb, "RGB")

            # For cubemap: both fovx and fovy must be 90 degrees
            # Use square output (min of width, height) to ensure proper cubemap coverage
            original_width, original_height = image.size
            cube_size = min(original_width, original_height)
            
            # Force fovx = fovy = 90 degrees for cubemap
            FovX = fovx  # Already 90 degrees (in radians)
            FovY = fovx  # Same as FovX for cubemap (must be 90 degrees)

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=cube_size, height=cube_size, depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos


def readCubemapInfo(path, white_background, depths, eval, face_name, extension=".png"):
    """
    Read cubemap scene info for a specific face direction.
    
    Args:
        path: Base path to the dataset
        white_background: Whether to use white background
        depths: Depth folder name
        eval: Whether in evaluation mode
        face_name: One of 'front', 'back', 'left', 'right', 'up', 'down'
        extension: Image file extension
    
    Returns:
        SceneInfo for the specified cubemap face
    """
    cubemap_folder = os.path.join(path, "cubemap")
    
    if not os.path.exists(cubemap_folder):
        raise FileNotFoundError(f"Cubemap folder not found: {cubemap_folder}")
    
    depths_folder = os.path.join(cubemap_folder, depths) if depths != "" else ""
    
    train_json = f"transforms_train_{face_name}.json"
    test_json = f"transforms_test_{face_name}.json"
    
    print(f"Reading Training Transforms for {face_name} face")
    train_cam_infos = readCamerasFromCubemapTransforms(cubemap_folder, path, train_json, depths_folder, white_background, False, extension)
    print(f"Reading Test Transforms for {face_name} face")
    test_cam_infos = readCamerasFromCubemapTransforms(cubemap_folder, path, test_json, depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos) if train_cam_infos else {"translate": np.zeros(3), "radius": 1.0}

    # Use the same point cloud as the original scene
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Try alternative location
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


def readAllCubemapInfo(path, white_background, depths, eval, extension=".png"):
    """
    Read all 6 cubemap faces at once.
    
    Returns:
        dict: {face_name: SceneInfo}
    """
    all_faces = {}
    for face_name in CUBEMAP_FACES:
        try:
            all_faces[face_name] = readCubemapInfo(path, white_background, depths, eval, face_name, extension)
            print(f"Loaded {face_name}: {len(all_faces[face_name].train_cameras)} train, {len(all_faces[face_name].test_cameras)} test")
        except Exception as e:
            print(f"Warning: Could not load {face_name} face: {e}")
    return all_faces