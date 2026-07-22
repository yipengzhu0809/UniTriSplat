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
import math
import json
import os
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def fisheye_params_from_fov(width, height, fov_x, fov_y):
    """Derive normalized anisotropic-equidistant camera parameters."""
    if width <= 0 or height <= 0:
        raise ValueError("Fisheye image dimensions must be positive")
    if not math.isfinite(fov_x) or not 0.0 < fov_x <= 360.0:
        raise ValueError("fisheye_fov_x must be in the range (0, 360]")
    if not math.isfinite(fov_y) or not 0.0 < fov_y <= 360.0:
        raise ValueError("fisheye_fov_y must be in the range (0, 360]")

    return {
        "fx": width / math.pi,
        "fy": height / math.pi,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "w_x": fov_x / 180.0,
        "w_y": fov_y / 180.0,
        "W": width,
        "H": height,
        "fov_x": float(fov_x),
        "fov_y": float(fov_y),
        "source": "FoV arguments",
    }


def _fisheye_fov_from_params(width, height, fx, fy, w_x, w_y):
    """Return axis FoVs in degrees for the anisotropic equidistant model."""
    return math.degrees(w_x * width / fx), math.degrees(w_y * height / fy)


def load_fisheye_params(source_path, width, height):
    """Load fisheye calibration from a transforms file and scale it to an image size."""
    required = (
        "fisheye_fx", "fisheye_fy", "fisheye_cx",
        "fisheye_cy", "fisheye_w_x", "fisheye_w_y",
    )
    for filename in ("transforms_train.json", "transforms_test.json"):
        path = os.path.join(source_path, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not all(key in metadata for key in required):
            continue

        source_width = float(metadata.get("fisheye_width", width))
        source_height = float(metadata.get("fisheye_height", height))
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Invalid fisheye image dimensions in {path}")

        scale_x = width / source_width
        scale_y = height / source_height
        fx = float(metadata["fisheye_fx"]) * scale_x
        fy = float(metadata["fisheye_fy"]) * scale_y
        cx = float(metadata["fisheye_cx"]) * scale_x
        cy = float(metadata["fisheye_cy"]) * scale_y
        w_x = float(metadata["fisheye_w_x"])
        w_y = float(metadata["fisheye_w_y"])
        if fx <= 0 or fy <= 0 or w_x <= 0 or w_y <= 0:
            raise ValueError(f"Invalid fisheye calibration in {path}")

        fov_x, fov_y = _fisheye_fov_from_params(width, height, fx, fy, w_x, w_y)
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "w_x": w_x, "w_y": w_y, "W": width, "H": height,
            "fov_x": fov_x, "fov_y": fov_y, "source": path,
        }
    return None


def resolve_fisheye_params(source_path, width, height, fov_x=-1.0, fov_y=-1.0):
    """Resolve explicit FoVs or fall back to calibration stored with the dataset."""
    has_fov_x = fov_x is not None and fov_x > 0
    has_fov_y = fov_y is not None and fov_y > 0
    if has_fov_x != has_fov_y:
        raise ValueError("fisheye_fov_x and fisheye_fov_y must be specified together")
    if has_fov_x:
        return fisheye_params_from_fov(width, height, fov_x, fov_y)

    params = load_fisheye_params(source_path, width, height)
    if params is None:
        raise ValueError(
            "Fisheye calibration was not found in transforms_train.json. "
            "Provide both --fisheye_fov_x and --fisheye_fov_y for centered "
            "normalized anisotropic-equidistant images."
        )
    return params
