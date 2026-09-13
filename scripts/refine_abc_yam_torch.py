"""Differentiable URDF motion calibration on CUDA (or CPU for validation).

Known joint states articulate fixed, corresponding surface samples. Optimize two
SE(3) base-to-camera transforms against measured optical flow. Uses DIS without
weights by default; --flow-backend raft uses pretrained torchvision RAFT on CUDA.
No Gaussian checkpoint is used or implied. MuJoCo only supplies visibility and
final mesh visualization; PyTorch computes differentiable projection and fitting.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from calibrate_abc_yam import ROOT, joints, project
from scipy.spatial.transform import Rotation
from yam_geometry import URDFRobot
from yam_renderer import YamRenderer


def flow_dis(a, b):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None
    )


def create_flow_backend(name, device):
    if name == "dis":
        return flow_dis
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    net = (
        raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval().to(device)
    )

    def raft(a, b):
        h, w = a.shape[:2]
        ph = (-h) % 8
        pw = (-w) % 8
        ims = [
            torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1]))
            .permute(2, 0, 1)
            .float()[None]
            .to(device)
            / 127.5
            - 1
            for im in [a, b]
        ]
        ims = [F.pad(im, (0, pw, 0, ph), mode="replicate") for im in ims]
        with torch.inference_mode():
            f = net(*ims)[-1][0, :, :h, :w]
        return f.permute(1, 2, 0).cpu().numpy()

    return raft


def skew(v):
    import torch

    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(
        *v.shape[:-1], 3, 3
    )


def rotate(delta, R0):
    import torch

    return torch.matrix_exp(skew(delta)) @ R0


def fit(out, device="cuda", steps=350, flow_backend="dis", pair_count=24):
    import torch
    import torch.nn.functional as F

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Use --device cpu for smoke tests."
        )
    if device == "cpu":
        torch.set_num_threads(4)
    z = dict(np.load(out / "prepared.npz"))
    base = json.load(open(out / "calibration.json"))
    H, W = z["frames"][0].shape[:2]
    K = z["K"].copy()
    K[:2] *= 0.5
    w, h = W // 2, H // 2
    prep = time.perf_counter()
    robot = URDFRobot(ROOT / "third_party/yam/i2rt/yam.urdf", nsample=240)
    # Exclude fingers: aperture-to-motor linkage varies by gripper revision.
    robot.samples = {
        k: v
        for k, v in robot.samples.items()
        if k not in ["base", "tip_left", "tip_right"]
    }
    pairs = np.linspace(5, len(z["frames"]) - 8, pair_count, dtype=int)
    flowfn = create_flow_backend(flow_backend, device)
    r = YamRenderer(ROOT / "third_party/yam/i2rt/yam.urdf", K, w, h)
    points0 = []
    points1 = []
    flows = []
    valids = []
    side_ids = []
    vis = []
    for i in pairs:
        im0 = cv2.resize(z["frames"][i], (w, h))
        im1 = cv2.resize(z["frames"][i + 1], (w, h))
        f = flowfn(im0, im1)
        back = flowfn(im1, im0)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        bx = xx + f[:, :, 0]
        by = yy + f[:, :, 1]
        bsample = cv2.remap(
            back,
            bx,
            by,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1e4,
        )
        confidence = (
            (np.linalg.norm(f + bsample, axis=2) < 1.5)
            & (bx >= 1)
            & (bx < w - 1)
            & (by >= 1)
            & (by < h - 1)
        )
        for side_id, side in enumerate(["left", "right"]):
            x = np.array(base["final_poses"][side])
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            p0 = robot.points(joints(z, side, i))
            p1 = robot.points(joints(z, side, i + 1))
            pc = p0 @ R.T + x[3:]
            uv = project(p0, x, K)
            uv1 = project(p1, x, K)
            u, v = np.round(uv).astype(int).T
            inside = (u >= 2) & (u < w - 2) & (v >= 2) & (v < h - 2)
            uc = u.clip(0, w - 1)
            vc = v.clip(0, h - 1)
            dep = r.depth(joints(z, side, i), x)
            visible = (
                inside
                & (np.abs(dep[vc, uc] - pc[:, 2]) < 0.015)
                & (np.linalg.norm(uv1 - uv, axis=1) > 0.75)
            )
            points0.append(p0)
            points1.append(p1)
            flows.append(f)
            valids.append(confidence)
            side_ids.append(side_id)
            vis.append(visible)
    r.close()
    print(
        "flow preprocessing",
        round(time.perf_counter() - prep, 2),
        "visible samples",
        int(np.sum(vis)),
        flush=True,
    )
    np.savez_compressed(
        out / "flow_diagnostics.npz",
        pair_indices=pairs,
        flows=np.array(flows[::2]),
        confidence=np.array(valids[::2]),
    )
    # Visual flow diagnostic independent of model: HSV hue is direction, brightness magnitude.
    previews = []
    for ix in np.linspace(0, len(pairs) - 1, 4, dtype=int):
        f = flows[2 * ix]
        mag, ang = cv2.cartToPolar(f[:, :, 0], f[:, :, 1])
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[:, :, 0] = ang * 90 / np.pi
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = np.minimum(mag * 15, 255)
        previews.append(
            np.concatenate(
                [
                    cv2.resize(z["frames"][pairs[ix]], (w, h)),
                    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
                ],
                axis=1,
            )
        )
    cv2.imwrite(str(out / "flow_contact.jpg"), np.concatenate(previews, axis=0))
    preprocessing = time.perf_counter() - prep

    def tensor(a):
        return torch.as_tensor(np.array(a), dtype=torch.float32, device=device)

    P0 = tensor(points0)
    P1 = tensor(points1)
    flow = tensor(flows).permute(0, 3, 1, 2)
    valid = tensor(valids)[:, None]
    visibility = tensor(vis)
    sid = torch.tensor(side_ids, device=device)
    R0 = tensor(
        [
            Rotation.from_rotvec(base["final_poses"][s][:3]).as_matrix()
            for s in ["left", "right"]
        ]
    )
    t0 = tensor([base["final_poses"][s][3:] for s in ["left", "right"]])
    intr = tensor(K)
    delta = torch.zeros((2, 6), device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.002)

    def positions():
        return rotate(delta[:, :3], R0), t0 + delta[:, 3:]

    def proj(p, R, t):
        pc = p @ R[sid].transpose(-1, -2) + t[sid, None, :]
        uv = pc[:, :, :2] / pc[:, :, 2:].clamp_min(0.05)
        return uv * intr.diag()[:2] + intr[:2, 2]

    def objective():
        R, t = positions()
        uv = proj(P0, R, t)
        uv1 = proj(P1, R, t)
        grid = torch.stack(
            [uv[:, :, 0] * 2 / (w - 1) - 1, uv[:, :, 1] * 2 / (h - 1) - 1], dim=-1
        )[:, :, None, :]
        real = F.grid_sample(flow, grid, align_corners=True).squeeze(-1).transpose(1, 2)
        confidence = (
            F.grid_sample(valid, grid, align_corners=True)
            .squeeze(1)
            .squeeze(-1)
            .detach()
        )
        weight = visibility * confidence
        error = uv1 - uv - real
        # Cauchy robust penalty tolerates occluded/incorrect flow; weights normalized per batch.
        per = torch.log1p((error**2).sum(-1) / 4)
        data_loss = (per * weight).sum() / weight.sum().clamp_min(1)
        prior = (
            0.003 * (delta[:, :3] ** 2).sum() / 0.15**2
            + 0.003 * (delta[:, 3:] ** 2).sum() / 0.08**2
        )
        return data_loss + prior, data_loss, weight.sum()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    if np.sum(vis) < 100:
        raise RuntimeError(
            "Insufficient visible moving robot samples for motion calibration"
        )
    sync()
    start = time.perf_counter()
    history = []
    with torch.no_grad():
        initial_loss = float(objective()[1])
    for it in range(steps):
        opt.zero_grad()
        loss, data_loss, count = objective()
        loss.backward()
        opt.step()
        if it == steps * 2 // 3:
            for g in opt.param_groups:
                g["lr"] *= 0.2
        if it % 10 == 0 or it == steps - 1:
            sync()
            with torch.no_grad():
                R, t = positions()
                rn = R.cpu().numpy()
                tn = t.cpu().numpy()
                poses = {
                    s: np.r_[Rotation.from_matrix(rn[j]).as_rotvec(), tn[j]].tolist()
                    for j, s in enumerate(["left", "right"])
                }
            history.append(
                {
                    "iteration": it,
                    "elapsed_seconds": time.perf_counter() - start,
                    "loss": float(data_loss),
                    "poses": poses,
                }
            )
            if it % 50 == 0:
                print(out.name, device, it, round(float(data_loss), 5), flush=True)
    sync()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        final_loss = float(objective()[1])
    result = dict(base)
    result.update(
        method="PyTorch differentiable URDF motion calibration after contour ICP; "
        + flow_backend
        + " optical flow",
        initial_poses=base["final_poses"],
        final_poses=history[-1]["poses"],
        history=history,
        iterations=steps,
        optimization_seconds=elapsed,
        coarse_optimization_seconds=base["optimization_seconds"],
        preprocessing_seconds=preprocessing,
        device=device,
        device_name=torch.cuda.get_device_name()
        if device == "cuda"
        else platform.processor(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        flow_backend=flow_backend,
        flow_pair_indices=pairs.tolist(),
        initial_flow_loss=initial_loss,
        final_flow_loss=final_loss,
    )
    (out / "calibration_coarse.json").write_text(json.dumps(base, indent=2))
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    print("complete", elapsed, initial_loss, final_loss, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--flow-backend", default="dis", choices=["dis", "raft"])
    p.add_argument("--pairs", type=int, default=24)
    a = p.parse_args()
    for out in (
        [a.episode] if a.episode else sorted((ROOT / "outputs/yam").glob("episode_*"))
    ):
        fit(out, a.device, a.steps, a.flow_backend, a.pairs)
