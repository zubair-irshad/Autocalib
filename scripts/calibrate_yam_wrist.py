"""ABC wrist adaptation of Cloak's low-intensity/low-variance silhouette method.

Algorithm reference: Stanford-TML/cloak, MIT, commit 174aa4e8eeec2141e7b5e02978dba9b1c9068bbf.
ABC aperture is 1=open (opposite DROID). Fits camera-to-URDF-gripper, not camera-to-base.
Reports failures and mask consistency rather than percentile-dropping the worst 1%.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from calibrate_abc_yam import ROOT, joints
from render_abc_yam import Video, label
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from yam_geometry import URDFRobot
from yam_renderer import YamRenderer

CLOAK_COMMIT = "174aa4e8eeec2141e7b5e02978dba9b1c9068bbf"


def pose_matrix(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:]
    return T


def matrix_pose(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def wrist_prior():
    # Cloak YAM prior is in attachment_site, whose axes are Ry(pi) from URDF gripper.
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(
        "xyz", [-28.5, -0.19, -90.36], degrees=True
    ).as_matrix()
    T[:3, 3] = [-0.09105, 0.03311, 0.02103]
    A = np.diag([-1.0, 1.0, -1.0, 1.0])
    return matrix_pose(A @ T)


def iou(a, b):
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def target_mask(frames, indices):
    gray = np.array(
        [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in indices], np.float32
    )
    med = np.median(gray, axis=0)
    std = np.std(gray, axis=0)
    h, w = med.shape
    roi = np.zeros((h, w), bool)
    roi[int(0.50 * h) :] = True
    dark = min(100.0, np.percentile(med[roi], 35))
    stable = min(25.0, np.percentile(std[roi], 35))
    mask = ((med < dark) & (std < stable) & roi).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area > max(15, h * w * 0.0005) and y + bh >= h - 4:
            clean[lab == k] = 1
    return clean.astype(bool), med, std


class WristRenderer:
    def __init__(self, K, w, h):
        self.robot = URDFRobot(ROOT / "third_party/yam/i2rt/yam.urdf", nsample=1)
        self.r = YamRenderer(ROOT / "third_party/yam/i2rt/yam.urdf", K, w, h)
        for i, bid in enumerate(self.r.model.geom_bodyid):
            name = self.r.model.body(bid).name
            if name not in ["gripper", "tip_left", "tip_right"]:
                self.r.model.geom_group[i] = 5
        self.r.opt.geomgroup[5] = 0

    def base_to_camera(self, q, cam_to_gripper):
        G = self.robot.transforms(q)["gripper"]
        return matrix_pose(np.linalg.inv(G @ pose_matrix(cam_to_gripper)))

    def mask(self, q, p):
        return self.r.depth(q, self.base_to_camera(q, p)) < 2

    def overlay(self, im, q, p):
        x = self.base_to_camera(q, p)
        mask = self.r.depth(q, x) < 2
        rgb = self.r.rgb(q, x, (255, 130, 30))
        out = im.copy()
        out[mask] = 0.45 * out[mask] + 0.55 * rgb[mask]
        cont, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, cont, -1, (30, 180, 255), 1)
        return out

    def close(self):
        self.r.close()


def fit(out, side, maxiter=250, videos=True):
    start_all = time.perf_counter()
    z = dict(np.load(out / "prepared.npz"))
    frames = z["frames"]
    h, w = frames[0].shape[:2]
    # Separate interleaved open-frame subsets for train and diagnostic validation masks.
    opened = np.flatnonzero(z[side + "_ee_state"][:, 0] >= 0.95)
    if len(opened) < 12:
        result = {
            "status": "rejected",
            "reason": "fewer_than_12_open_frames",
            "open_frames": len(opened),
        }
        (out / "wrist_calibration.json").write_text(json.dumps(result, indent=2))
        return result
    selected = opened[np.linspace(0, len(opened) - 1, min(80, len(opened)), dtype=int)]
    train = selected[::2]
    test = selected[1::2]
    small = np.array([cv2.resize(im, (w // 2, h // 2)) for im in frames])
    target, med, std = target_mask(small, train)
    held, _, _ = target_mask(small, test)
    K = z["K"].copy()
    K[:2] *= 0.5
    r = WristRenderer(K, w // 2, h // 2)
    q = joints(z, side, int(train[0]))
    q[
        6:
    ] = (
        -0.04695
    )  # open reference: jaw mismatch in selected .95..1 range is below 2.4 mm.
    init = wrist_prior()
    history = []
    start = time.perf_counter()
    evals = 0

    def loss(p):
        nonlocal evals
        evals += 1
        # A conservative mount prior avoids implausible poses fitting background speckles.
        dt = np.linalg.norm(p[3:] - init[3:])
        dr = (
            Rotation.from_rotvec(p[:3]) * Rotation.from_rotvec(init[:3]).inv()
        ).magnitude()
        if dt > 0.12 or dr > np.deg2rad(65):
            return 2.0 + dt + dr
        return 1 - iou(r.mask(q, p), target)

    def callback(p):
        history.append(
            {
                "iteration": len(history),
                "elapsed_seconds": time.perf_counter() - start,
                "pose": p.tolist(),
                "iou": 1 - loss(p),
            }
        )

    initial_mask = r.mask(q, init)
    initial_iou = iou(initial_mask, target)
    steps = np.array([0.035] * 3 + [0.005] * 3)
    simplex = np.vstack([init] + [init + e * steps for e in np.eye(6)])
    # Cloak's DROID mean is too far from ABC mounts for one local simplex.
    # Search a bounded camera-mount neighborhood, then refine the best candidate.
    candidates = []
    Rseed = Rotation.from_rotvec(init[:3])
    for tx in [0.05, 0.08, 0.11, 0.14]:
        for ty in [-0.025, 0, 0.025]:
            for tz in [-0.045, -0.015, 0.015]:
                for pitch in [-0.4, -0.2, 0, 0.2, 0.4]:
                    v = init.copy()
                    v[3:] = [tx, ty, tz]
                    v[:3] = (Rseed * Rotation.from_rotvec([pitch, 0, 0])).as_rotvec()
                    candidates.append((loss(v), v))
    score, start_pose = min(candidates, key=lambda a: a[0])
    history.append(
        {
            "iteration": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "pose": start_pose.tolist(),
            "iou": 1 - score,
        }
    )
    simplex = np.vstack([start_pose] + [start_pose + e * steps for e in np.eye(6)])
    sol = minimize(
        loss,
        start_pose,
        method="Nelder-Mead",
        callback=callback,
        options={
            "maxiter": maxiter,
            "initial_simplex": simplex,
            "xatol": 1e-4,
            "fatol": 1e-4,
            "adaptive": True,
        },
    )
    elapsed = time.perf_counter() - start
    final = r.mask(q, sol.x)
    fit_iou = iou(final, target)
    held_iou = iou(final, held)
    agreement = iou(target, held)
    area = float(target.mean())
    reasons = []
    if area < 0.005 or area > 0.25:
        reasons.append("implausible_target_area")
    if agreement < 0.55:
        reasons.append("unstable_temporal_mask")
    if fit_iou < 0.6:
        reasons.append("low_silhouette_iou")
    if held_iou < 0.55:
        reasons.append("low_held_out_iou")
    result = {
        "status": "accepted_for_review" if not reasons else "rejected",
        "reasons": reasons,
        "method": "Cloak-inspired ABC/YAM wrist silhouette calibration",
        "cloak_commit": CLOAK_COMMIT,
        "side": side,
        "camera_to_frame": "I2RT v1 URDF gripper",
        "pose_convention": "rotvec xyz followed by translation xyz; maps OpenCV camera coordinates into URDF gripper frame",
        "initial_pose": init.tolist(),
        "camera_to_gripper": sol.x.tolist(),
        "T_gripper_camera": pose_matrix(sol.x).tolist(),
        "initial_iou": initial_iou,
        "train_iou": fit_iou,
        "held_out_mask_iou": held_iou,
        "train_test_mask_iou": agreement,
        "target_area_fraction": area,
        "open_frame_count": len(opened),
        "train_frames": train.tolist(),
        "held_out_frames": test.tolist(),
        "optimization_seconds": elapsed,
        "preprocess_seconds": start - start_all,
        "coarse_search_candidates": len(candidates),
        "evaluations": evals,
        "iterations": int(sol.nit),
        "solver_success": bool(sol.success),
        "history": history,
        "note": "Silhouette diagnostics are not ground-truth extrinsic accuracy; validation shares the same gripper opening and cannot fully resolve pose ambiguity.",
    }
    (out / "wrist_calibration.json").write_text(json.dumps(result, indent=2))
    diag = np.concatenate(
        [
            cv2.cvtColor(med.astype(np.uint8), cv2.COLOR_GRAY2BGR),
            np.repeat((target * 255).astype(np.uint8)[:, :, None], 3, axis=2),
            np.repeat((final * 255).astype(np.uint8)[:, :, None], 3, axis=2),
        ],
        axis=1,
    )
    cv2.imwrite(str(out / "mask_diagnostic.jpg"), diag)
    r.close()
    r = WristRenderer(z["K"], w, h)
    idx = int(train[0])
    cv2.imwrite(
        str(out / "final_overlay.jpg"),
        r.overlay(frames[idx], joints(z, side, idx), sol.x),
    )
    if videos:
        v = Video(out / "optimization.mp4", w, h + 64, 12)
        states = (
            [
                {
                    "pose": init.tolist(),
                    "iteration": -1,
                    "elapsed_seconds": 0,
                    "iou": initial_iou,
                }
            ]
            + history[:: max(1, len(history) // 90)]
            + [
                {
                    "pose": sol.x.tolist(),
                    "iteration": sol.nit,
                    "elapsed_seconds": elapsed,
                    "iou": fit_iou,
                }
            ]
        )
        for a in states:
            im = label(
                r.overlay(frames[idx], joints(z, side, idx), np.array(a["pose"])),
                f"{side} wrist | silhouette iteration {a['iteration'] + 1}",
                f"IoU {a['iou']:.3f} | {a['elapsed_seconds']:.2f}s | {result['status']}",
            )
            v.write(im)
        for _ in range(18):
            v.write(im)
        v.close()
        v = Video(out / "final_overlay.mp4", w * 2, h + 64, 6)
        for i in range(len(frames)):
            a = label(
                frames[i], f"{side} wrist: recorded", f"{i / 2:.1f}s | playback 3x"
            )
            b = label(
                r.overlay(frames[i], joints(z, side, i), sol.x),
                "YAM URDF gripper overlay",
                f"{result['status']} | open-frame fit, all-aperture playback",
            )
            v.write(np.concatenate([a, b], axis=1))
        v.close()
    r.close()
    print(
        out,
        {
            k: result[k]
            for k in [
                "status",
                "optimization_seconds",
                "initial_iou",
                "train_iou",
                "held_out_mask_iou",
                "train_test_mask_iou",
            ]
        },
        flush=True,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path)
    p.add_argument("--maxiter", type=int, default=250)
    p.add_argument("--skip-videos", action="store_true")
    a = p.parse_args()
    for ep in (
        [a.episode] if a.episode else sorted((ROOT / "outputs/yam").glob("episode_*"))
    ):
        for side in ["left", "right"]:
            out = ep / ("wrist_" + side)
            if (out / "preparation.json").exists():
                fit(out, side, a.maxiter, not a.skip_videos)
