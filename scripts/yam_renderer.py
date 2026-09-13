"""MuJoCo rendering of the original I2RT URDF visual meshes."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class YamRenderer:
    def __init__(self, urdf, K, width, height):
        self.K = K.copy()
        self.w = width
        self.h = height
        self.xscale = float(K[1, 1] / K[0, 0])
        self.render_w = int(np.ceil(width * self.xscale))
        self.render_K = K.copy()
        self.render_K[0, :] *= self.xscale
        urdf = Path(urdf).resolve()
        root = ET.parse(urdf).getroot()
        ext = ET.SubElement(root, "mujoco")
        ET.SubElement(
            ext,
            "compiler",
            balanceinertia="true",
            discardvisual="false",
            fusestatic="false",
            strippath="false",
        )
        for mesh in root.iter("mesh"):
            mesh.set(
                "filename",
                str(urdf.parent / mesh.get("filename").replace("package://", "")),
            )
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td) / "converted.xml"
            mujoco.mj_saveLastXML(str(temp), m)
            r = ET.parse(temp).getroot()
        visual = ET.SubElement(r, "visual")
        ET.SubElement(
            visual, "global", offwidth=str(self.render_w), offheight=str(height)
        )
        ET.SubElement(visual, "map", znear=".01", zfar="10")
        ET.SubElement(
            visual,
            "headlight",
            ambient=".6 .6 .6",
            diffuse=".5 .5 .5",
            specular=".1 .1 .1",
        )
        wb = r.find("worldbody")
        ET.SubElement(
            wb,
            "camera",
            name="calib",
            pos="0 0 0",
            xyaxes="1 0 0 0 -1 0",
            fovy=str(np.degrees(2 * np.arctan(height / (2 * K[1, 1])))),
        )
        ET.SubElement(
            wb,
            "light",
            pos="0 -1 0",
            dir="0 0 1",
            directional="true",
            castshadow="false",
            diffuse=".6 .6 .6",
        )
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, self.render_w)
        self.root = 1
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.joint_ids = [self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 9)]
        self.xml = ET.tostring(r, encoding="unicode")

    def setup(self, q, x):
        self.data.qpos[self.joint_ids] = q
        self.model.body_pos[self.root] = x[3:6]
        quat = Rotation.from_rotvec(x[:3]).as_quat()
        self.model.body_quat[self.root] = quat[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="calib", scene_option=self.opt)
        # Exact OpenCV pinhole intrinsics, with camera axes +x right, +y down, +z forward.
        for c in self.renderer.scene.camera:
            n = c.frustum_near
            c.frustum_top = self.K[1, 2] * n / self.K[1, 1]
            c.frustum_bottom = -(self.h - self.K[1, 2]) * n / self.K[1, 1]
            c.frustum_center = (
                (self.render_w / 2 - self.render_K[0, 2]) * n / self.render_K[0, 0]
            )

    def resample(self, im, nearest=False):
        if abs(self.xscale - 1) < 1e-6:
            return im
        yy, xx = np.mgrid[: self.h, : self.w].astype(np.float32)
        return cv2.remap(
            im, xx * self.xscale, yy, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        )

    def depth(self, q, x):
        self.setup(q, x)
        self.renderer.enable_depth_rendering()
        im = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return self.resample(im, True)

    def rgb(self, q, x, color):
        self.model.geom_rgba[:, :3] = np.array(color)[None, :] / 255
        self.model.geom_rgba[:, 3] = 1
        self.setup(q, x)
        return self.resample(self.renderer.render().copy()[:, :, ::-1])

    def close(self):
        self.renderer.close()
