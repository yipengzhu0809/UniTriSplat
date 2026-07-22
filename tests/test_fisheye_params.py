import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("torch", types.ModuleType("torch"))

from utils.graphics_utils import resolve_fisheye_params


class FisheyeParameterTests(unittest.TestCase):
    def test_explicit_fov_uses_normalized_parameterization(self):
        params = resolve_fisheye_params("/missing", 1024, 1024, 155.0, 155.0)
        self.assertAlmostEqual(params["fx"], 1024 / math.pi)
        self.assertAlmostEqual(params["fy"], 1024 / math.pi)
        self.assertAlmostEqual(params["w_x"], 155 / 180)
        self.assertAlmostEqual(params["w_y"], 155 / 180)
        self.assertEqual(params["source"], "FoV arguments")

    def test_metadata_preserves_calibration_and_scales_intrinsics(self):
        metadata = {
            "fisheye_fx": 744.7158292834381,
            "fisheye_fy": 776.9179801361099,
            "fisheye_cx": 878.6424154066442,
            "fisheye_cy": 585.380971294785,
            "fisheye_w_x": 1.0,
            "fisheye_w_y": 1.0,
            "fisheye_width": 1752,
            "fisheye_height": 1168,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transforms_train.json"
            path.write_text(json.dumps(metadata))
            params = resolve_fisheye_params(directory, 876, 584)

        self.assertAlmostEqual(params["fx"], metadata["fisheye_fx"] / 2)
        self.assertAlmostEqual(params["fy"], metadata["fisheye_fy"] / 2)
        self.assertAlmostEqual(params["cx"], metadata["fisheye_cx"] / 2)
        self.assertAlmostEqual(params["cy"], metadata["fisheye_cy"] / 2)
        self.assertEqual(params["w_x"], 1.0)
        self.assertEqual(params["w_y"], 1.0)
        self.assertAlmostEqual(
            params["fov_x"],
            math.degrees(metadata["fisheye_width"] / metadata["fisheye_fx"]),
        )
        self.assertAlmostEqual(
            params["fov_y"],
            math.degrees(metadata["fisheye_height"] / metadata["fisheye_fy"]),
        )

    def test_explicit_fov_overrides_metadata(self):
        metadata = {
            "fisheye_fx": 700.0,
            "fisheye_fy": 700.0,
            "fisheye_cx": 800.0,
            "fisheye_cy": 500.0,
            "fisheye_w_x": 1.0,
            "fisheye_w_y": 1.0,
            "fisheye_width": 1600,
            "fisheye_height": 1000,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transforms_train.json"
            path.write_text(json.dumps(metadata))
            params = resolve_fisheye_params(directory, 1000, 1000, 120.0, 100.0)

        self.assertAlmostEqual(params["fx"], 1000 / math.pi)
        self.assertAlmostEqual(params["fy"], 1000 / math.pi)
        self.assertAlmostEqual(params["w_x"], 120 / 180)
        self.assertAlmostEqual(params["w_y"], 100 / 180)

    def test_axis_fovs_must_be_provided_together(self):
        with self.assertRaisesRegex(ValueError, "must be specified together"):
            resolve_fisheye_params("/missing", 1000, 1000, 155.0, -1.0)

    def test_missing_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "calibration was not found"):
                resolve_fisheye_params(directory, 1000, 1000)


if __name__ == "__main__":
    unittest.main()
