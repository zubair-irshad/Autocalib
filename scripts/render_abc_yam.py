"""Render recorded solver iterates and final URDF overlays, plus validation metrics."""

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from calibrate_abc_yam import ROOT, contour_points, image_edges, joints
from yam_renderer import YamRenderer


class Video:
    def __init__(self, path, w, h, fps):
        self.p = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, im):
        self.p.stdin.write(np.ascontiguousarray(im).tobytes())

    def close(self):
        self.p.stdin.close()
        assert self.p.wait() == 0


def overlay(r, z, i, poses, alpha=0.55):
    im = z["frames"][i].copy()
    depths = []
    rgbs = []
    for side, col in [("left", (30, 190, 255)), ("right", (255, 130, 40))]:
        q = joints(z, side, i)
        x = np.array(poses[side])
        depths.append(r.depth(q, x))
        rgbs.append(r.rgb(q, x, col))
    depths = np.array(depths)
    for j in range(2):
        mask = (depths[j] < 4) & (depths[j] <= depths[1 - j])
        im[mask] = ((1 - alpha) * im[mask] + alpha * rgbs[j][mask]).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            im, contours, -1, [(255, 200, 30), (40, 150, 255)][j], 1, cv2.LINE_AA
        )
    return im


def label(im, top, bottom=""):
    out = np.zeros((im.shape[0] + 64, im.shape[1], 3), np.uint8)
    out[64:] = im
    cv2.putText(
        out,
        top,
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        bottom,
        (16, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (195, 205, 215),
        1,
        cv2.LINE_AA,
    )
    return out


def render(out, quick=False):
    start = time.perf_counter()
    z = dict(np.load(out / "prepared.npz"))
    cal = json.load(open(out / "calibration.json"))
    H, W = z["frames"][0].shape[:2]
    r = YamRenderer(ROOT / "third_party/yam/i2rt/yam.urdf", z["K"], W, H)
    indices = np.linspace(0, len(z["frames"]) - 1, 6, dtype=int)
    tiles = []
    for i in indices:
        im = overlay(r, z, i, cal["final_poses"])
        cv2.imwrite(str(out / f"overlay_{i:04d}.jpg"), im)
        tile = cv2.resize(im, (480, 300))
        cv2.putText(
            tile, f"Frame {i} | {i / 5:.1f}s", (12, 24), 0, 0.6, (255, 255, 255), 2
        )
        tiles.append(tile)
    cv2.imwrite(
        str(out / "final_contact.jpg"),
        np.concatenate(
            [np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)],
            axis=0,
        ),
    )
    if quick:
        r.close()
        return
    # Hold a representative actual frame fixed while replaying recorded optimization states.
    idx = cal["training_frame_indices"][4]
    v = Video(out / "optimization.mp4", W, H + 64, 10)
    coarse = (
        json.load(open(out / "calibration_coarse.json"))
        if (out / "calibration_coarse.json").exists()
        else None
    )
    hist = []
    if coarse:
        hist = [
            {
                "iteration": -1,
                "elapsed_seconds": 0,
                "poses": coarse["initial_poses"],
                "stage": "Contour initialization",
            }
        ] + [dict(a, stage="Contour ICP") for a in coarse["history"]]
    offset = coarse["optimization_seconds"] if coarse else 0
    hist += [
        dict(
            a,
            elapsed_seconds=a["elapsed_seconds"] + offset,
            stage="Motion refinement" if coarse else "Contour ICP",
        )
        for a in cal["history"]
    ]
    for rec in hist:
        im = overlay(r, z, idx, rec["poses"])
        im = label(
            im,
            f"{rec['stage']} | {out.name[8:16]} | iteration {rec['iteration'] + 1}",
            f"Actual solver state | elapsed {rec['elapsed_seconds']:.2f}s | known K and joints | actual optimizer iterations",
        )
        for _ in range(3):
            v.write(im)
    for _ in range(20):
        v.write(im)
    v.close()
    v = Video(out / "final_overlay.mp4", W * 2, H + 64, 10)
    # All sampled frames (5 Hz), played at 2x; source and overlay remain synchronized.
    ts = z["timestamps_ns"]
    for i in range(len(z["frames"])):
        left = label(
            z["frames"][i],
            "Recorded top-left camera",
            f"{(ts[i] - ts[0]) / 1e9:.2f}s | frame {i} | playback 2x",
        )
        right = label(
            overlay(r, z, i, cal["final_poses"]),
            "Fitted YAM URDF | cyan: left, orange: right",
            "One fixed transform per arm for the full episode | 55% mesh opacity",
        )
        v.write(np.concatenate([left, right], axis=1))
        if i % 100 == 0:
            print(out.name, "render", i, flush=True)
    v.close()
    r.close()
    # Held-out contour distances, an image-edge proxy rather than ground-truth pose error.
    K = z["K"].copy()
    K[:2] *= 0.5
    r = YamRenderer(ROOT / "third_party/yam/i2rt/yam.urdf", K, W // 2, H // 2)
    used = set(cal["training_frame_indices"])
    used.update(cal.get("flow_pair_indices", []))
    used.update(i + 1 for i in cal.get("flow_pair_indices", []))
    test = [
        int(i)
        for i in np.linspace(4, len(z["frames"]) - 5, 24, dtype=int)
        if int(i) not in used
    ]
    scores = {}
    if coarse:
        cal["initial_poses"] = coarse["initial_poses"]
    for state in ["initial_poses", "final_poses"]:
        scores[state] = {}
        for side in ["left", "right"]:
            errors = []
            for i in test:
                _, dist, _, _, _ = image_edges(
                    cv2.resize(z["frames"][i], (W // 2, H // 2))
                )
                _, uv, _ = contour_points(
                    r.depth(joints(z, side, i), np.array(cal[state][side])), K
                )
                u, y = uv.T
                errors.extend(dist[y, u].tolist())
            scores[state][side] = {
                "mean_edge_distance_px_at_960": float(2 * np.mean(errors)),
                "median_edge_distance_px_at_960": float(2 * np.median(errors)),
                "within_5px_fraction": float(np.mean(np.array(errors) <= 2.5)),
            }
    r.close()
    validation = {
        "held_out_frames": test,
        "metrics": scores,
        "render_and_validation_seconds": time.perf_counter() - start,
        "caveat": "Nearest-image-edge distance is a proxy; clutter and internal edges can produce low scores for an incorrect pose. No ground-truth extrinsics.",
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    print(validation, flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    for out in (
        [a.episode] if a.episode else sorted((ROOT / "outputs/yam").glob("episode_*"))
    ):
        render(out, a.quick)
