"""Numerical checks for pose conventions and gradients used by the CUDA path."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import torch
from calibrate_abc_yam import ROOT, joints, project, seed
from refine_abc_yam_torch import rotate
from scipy.spatial.transform import Rotation
from yam_geometry import URDFRobot


class GeometryTests(unittest.TestCase):
    def test_urdf_fk_matches_mujoco(self):
        import xml.etree.ElementTree as ET

        import mujoco

        path = ROOT / "third_party/yam/i2rt/yam.urdf"
        robot = URDFRobot(path, nsample=10)
        tree = ET.parse(path).getroot()
        ext = ET.SubElement(tree, "mujoco")
        ET.SubElement(
            ext,
            "compiler",
            discardvisual="false",
            fusestatic="false",
            strippath="false",
        )
        for mesh in tree.iter("mesh"):
            mesh.set("filename", str(path.parent / mesh.get("filename")))
        m = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
        d = mujoco.MjData(m)
        rng = np.random.default_rng(2)
        for _ in range(5):
            q = np.r_[rng.uniform(-0.4, 0.4, 6), [-0.015, -0.015]]
            for j, v in zip(robot.names, q):
                d.qpos[m.joint(j).qposadr[0]] = v
            mujoco.mj_forward(m, d)
            for name, T in robot.transforms(q).items():
                np.testing.assert_allclose(T[:3, 3], d.body(name).xpos, atol=1e-7)
                np.testing.assert_allclose(
                    T[:3, :3], d.body(name).xmat.reshape(3, 3), atol=1e-7
                )

    def test_abc_open_aperture_separates_fingers(self):
        robot = URDFRobot(ROOT / "third_party/yam/i2rt/yam.urdf", nsample=300)
        widths = []
        for aperture in [0.0, 1.0]:
            q = joints(
                {
                    "left_ee_state": np.array([[aperture]]),
                    "left_arm_state": np.zeros((1, 6)),
                },
                "left",
                0,
            )
            T = robot.transforms(q)
            G = np.linalg.inv(T["gripper"])
            centers = []
            for name in ["tip_left", "tip_right"]:
                X = G @ T[name]
                centers.append((robot.samples[name] @ X[:3, :3].T + X[:3, 3]).mean(0))
            widths.append(abs(centers[0][1] - centers[1][1]))
        self.assertGreater(widths[1] - widths[0], 0.08)

    def test_rotation_gradient_at_zero(self):
        r0 = torch.tensor(
            Rotation.from_rotvec(seed("left")[:3]).as_matrix(), dtype=torch.double
        )
        d = torch.zeros(3, dtype=torch.double, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(lambda v: rotate(v, r0), (d,), atol=1e-5, eps=1e-6)
        )

    def test_torch_projection_matches_opencv(self):
        import cv2

        points = np.random.default_rng(3).normal(size=(30, 3)) * 0.1
        x = seed("left")
        K = np.array([[380.0, 0, 491.0], [0, 380.0, 312.0], [0, 0, 1.0]])
        cv, _ = cv2.projectPoints(points, x[:3], x[3:], K, None)
        np.testing.assert_allclose(project(points, x, K), cv[:, 0], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
