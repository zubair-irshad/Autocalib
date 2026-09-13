"""CPU mesh calibration for ABC-130k, with recorded solver iterates and timings.

Uses known intrinsics/joint angles, a tabletop camera-pose prior, and multi-frame
image edges. This is a separate mesh baseline, not the CUDA Gaussian pipeline.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yam_geometry import URDFRobot

ROOT = Path(__file__).resolve().parents[1]


def project(p, x, K):
    c = p @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]
    uv = c[:, :2] / np.maximum(c[:, 2:], 0.05)
    return uv * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]


def seed(side):
    R = np.array(
        [[0.0, -1.0, 0.0], [-(2**-0.5), 0.0, -(2**-0.5)], [2**-0.5, 0.0, -(2**-0.5)]]
    )
    return np.r_[
        Rotation.from_matrix(R).as_rotvec(),
        [-0.33 if side == "left" else 0.40, 0.53, 0.85],
    ]


def joints(z, side, i):
    a = float(z[side + "_ee_state"][i, 0])
    return np.r_[z[side + "_arm_state"][i], [-0.04695 * a] * 2]


def preview(z, robot, out):
    im = z["frames"][0].copy()
    K = z["K"]
    for side, color in [("left", (255, 140, 0)), ("right", (0, 100, 255))]:
        q = joints(z, side, 0)
        p = robot.points(q)
        uv = project(p, seed(side), K).astype(int)
        for u, v in uv:
            if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
                cv2.circle(im, (u, v), 1, color, -1)
    cv2.imwrite(str(out / "seed.jpg"), im)


def image_edges(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    edge = cv2.Canny(gray, 40, 100) > 0
    from scipy.ndimage import distance_transform_edt

    dist, indices = distance_transform_edt(~edge, return_indices=True)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return edge, dist, indices, gx, gy


def contour_points(depth, K):
    mask = (depth < 4).astype(np.uint8)
    cont = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
    y, u = np.nonzero(cont)
    good = (u > 3) & (u < depth.shape[1] - 4) & (y > 3) & (y < depth.shape[0] - 4)
    u = u[good]
    y = y[good]
    step = max(1, len(u) // 550)
    u = u[::step]
    y = y[::step]
    z = depth[y, u]
    p = np.c_[(u - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z]
    gx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0)[y, u]
    gy = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1)[y, u]
    return p, np.c_[u, y], np.c_[gx, gy]


def fit(z, out, iterations=30):
    from yam_renderer import YamRenderer

    begin = time.perf_counter()
    frames = z["frames"]
    H, W = frames[0].shape[:2]
    scale = 0.5
    K = z["K"].copy()
    K[:2] *= scale
    # Deterministic time-spread training split; remaining frames are unused by the solver.
    train = np.linspace(8, len(frames) - 9, 16, dtype=int)
    small = [cv2.resize(frames[i], (W // 2, H // 2)) for i in train]
    targets = [image_edges(im) for im in small]
    renderer = YamRenderer(ROOT / "third_party/yam/i2rt/yam.urdf", K, W // 2, H // 2)
    preprocessing = time.perf_counter() - begin
    start = time.perf_counter()
    records = []
    poses = {s: seed(s) for s in ["left", "right"]}
    for it in range(iterations):
        costs = {}
        counts = {}
        for side in ["left", "right"]:
            x = poses[side]
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            points = []
            observed = []
            errors = []
            for idx, target in zip(train, targets):
                dep = renderer.depth(joints(z, side, idx), x)
                pc, uv, norm = contour_points(dep, K)
                u, v = uv.T
                _, dist, indices, gx, gy = target
                ty = indices[0, v, u]
                tx = indices[1, v, u]
                ng = np.c_[gx[ty, tx], gy[ty, tx]]
                align = np.abs(np.sum(norm * ng, axis=1)) / (
                    np.linalg.norm(norm, axis=1) * np.linalg.norm(ng, axis=1) + 1e-6
                )
                keep = (dist[v, u] < max(6, 22 - it * 0.6)) & (align > 0.5)
                # Visibility and robust correspondence gating; fit both arms independently.
                points.append((pc[keep] - x[3:]) @ R)
                observed.append(np.c_[tx[keep], ty[keep]])
                errors.extend(dist[v, u].tolist())
            p = np.concatenate(points)
            obs = np.concatenate(observed)
            counts[side] = len(p)

            def residual(v):
                return (project(p, v, K) - obs).ravel()

            sol = least_squares(
                residual, x, loss="soft_l1", f_scale=2, max_nfev=20, diff_step=1e-4
            )
            poses[side] = sol.x
            costs[side] = float(np.mean(np.minimum(errors, 25)))
        rec = {
            "iteration": it,
            "elapsed_seconds": time.perf_counter() - start,
            "edge_distance_px_480": costs,
            "correspondences": counts,
            "poses": {s: x.tolist() for s, x in poses.items()},
        }
        records.append(rec)
        print(
            out.name,
            it,
            {s: round(v, 3) for s, v in costs.items()},
            round(rec["elapsed_seconds"], 2),
            flush=True,
        )
    renderer.close()
    seconds = time.perf_counter() - start
    result = {
        "method": "URDF multi-frame contour ICP; tabletop pose prior; known K and joint states; no Gaussian model",
        "episode": out.name,
        "training_frame_indices": train.tolist(),
        "optimization_resolution": [W // 2, H // 2],
        "iterations": iterations,
        "optimization_seconds": seconds,
        "preprocessing_seconds": preprocessing,
        "initial_poses": {s: seed(s).tolist() for s in poses},
        "final_poses": {s: x.tolist() for s, x in poses.items()},
        "history": records,
        "transform_convention": "p_camera = R(rotvec) @ p_arm_base + t; camera x right, y down, z forward",
    }
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--iterations", type=int, default=30)
    a = p.parse_args()
    for out in (
        [a.episode] if a.episode else sorted((ROOT / "outputs/yam").glob("episode_*"))
    ):
        if not (out / "preparation.json").exists():
            continue
        z = dict(np.load(out / "prepared.npz"))
        if a.preview:
            preview(z, URDFRobot(ROOT / "third_party/yam/i2rt/yam.urdf"), out)
        else:
            fit(z, out, a.iterations)
