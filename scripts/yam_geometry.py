"""URDF forward kinematics and mesh samples, independent of CUDA/trained splats."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def xyz(s):
    return np.fromstring(s, sep=" ")


def origin(el):
    T = np.eye(4)
    if el is not None:
        T[:3, 3] = xyz(el.get("xyz", "0 0 0"))
        T[:3, :3] = Rotation.from_euler("xyz", xyz(el.get("rpy", "0 0 0"))).as_matrix()
    return T


class URDFRobot:
    def __init__(self, path, nsample=350):
        self.path = Path(path)
        root = ET.parse(path).getroot()
        self.joints = []
        self.meshes = {}
        self.samples = {}
        for j in root.findall("joint"):
            self.joints.append(
                (
                    j.get("name"),
                    j.get("type"),
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                    origin(j.find("origin")),
                    xyz(j.find("axis").get("xyz", "1 0 0"))
                    if j.find("axis") is not None
                    else np.array([1, 0, 0]),
                )
            )
        children = {j[3] for j in self.joints}
        self.root = next(
            l.get("name") for l in root.findall("link") if l.get("name") not in children
        )
        self.names = sorted([j[0] for j in self.joints if j[1] != "fixed"])
        for l in root.findall("link"):
            parts = []
            for v in l.findall("visual"):
                m = v.find("geometry/mesh")
                if m is None:
                    continue
                path = self.path.parent / m.get("filename").replace("package://", "")
                mesh = trimesh.load(path, force="mesh", process=False)
                mesh.apply_scale(xyz(m.get("scale", "1 1 1")))
                mesh.apply_transform(origin(v.find("origin")))
                parts.append(mesh)
            if parts:
                mesh = trimesh.util.concatenate(parts)
                self.meshes[l.get("name")] = mesh
                self.samples[l.get("name")] = trimesh.sample.sample_surface(
                    mesh, nsample, seed=42
                )[0]

    def transforms(self, q):
        values = dict(zip(self.names, q))
        T = {self.root: np.eye(4)}
        remaining = list(self.joints)
        while remaining:
            for j in remaining[:]:
                name, typ, parent, child, O, axis = j
                if parent not in T:
                    continue
                M = np.eye(4)
                v = values.get(name, 0)
                if typ in ["revolute", "continuous"]:
                    M[:3, :3] = Rotation.from_rotvec(axis * v).as_matrix()
                elif typ == "prismatic":
                    M[:3, 3] = axis * v
                T[child] = T[parent] @ O @ M
                remaining.remove(j)
        return T

    def points(self, q, include_base=True):
        T = self.transforms(q)
        return np.concatenate(
            [
                p @ T[k][:3, :3].T + T[k][:3, 3]
                for k, p in self.samples.items()
                if include_base or k != self.root
            ]
        )

    def centers(self, q):
        T = self.transforms(q)
        return np.array([T[j[3]][:3, 3] for j in self.joints[:6]])
