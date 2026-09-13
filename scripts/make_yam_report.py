"""Assemble measured timings and quality diagnostics without extrapolating GPU performance."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "outputs/yam"
rows = []
quality = []
for ep in sorted(out.glob("episode_*")):
    cal = json.load(open(ep / "calibration.json"))
    prep = json.load(open(ep / "preparation.json"))
    val = json.load(open(ep / "validation.json"))
    wrist = {
        s: json.load(open(ep / ("wrist_" + s) / "wrist_calibration.json"))
        for s in ["left", "right"]
    }
    row = {
        "episode": ep.name,
        "top_contour_s": cal["coarse_optimization_seconds"],
        "top_motion_s": cal["optimization_seconds"],
        "top_total_s": cal["coarse_optimization_seconds"] + cal["optimization_seconds"],
        "left_wrist_s": wrist["left"]["optimization_seconds"],
        "right_wrist_s": wrist["right"]["optimization_seconds"],
        "top_decode_and_cache_s": prep["decode_seconds"],
        "top_flow_preprocess_s": cal["preprocessing_seconds"],
        "top_render_and_validation_s": val["render_and_validation_seconds"],
    }
    for s in ["left", "right"]:
        row[s + "_wrist_decode_and_cache_s"] = json.load(
            open(ep / ("wrist_" + s) / "preparation.json")
        )["decode_seconds"]
    rows.append(row)
    quality.append(
        {
            "episode": ep.name,
            "top": val["metrics"],
            "flow_loss": [cal["initial_flow_loss"], cal["final_flow_loss"]],
            "wrist": {
                s: {
                    k: wrist[s][k]
                    for k in [
                        "status",
                        "initial_iou",
                        "train_iou",
                        "held_out_mask_iou",
                        "train_test_mask_iou",
                    ]
                }
                for s in wrist
            },
        }
    )
(out / "timing_summary.json").write_text(
    json.dumps(
        {
            "hardware": "Apple Silicon Mac, CPU PyTorch 2.5.1; macOS MuJoCo graphics",
            "notes": "Single measured runs; not GPU benchmarks. Optimization includes solver work and in-loop rendering; excludes decode/cache construction, model initialization, flow preprocessing and output video generation. Some runs had concurrent local work.",
            "results": rows,
            "quality": quality,
        },
        indent=2,
    )
)
with (out / "timings.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
text = """# YAM / ABC-130k demonstration

Two local flower-arranging episodes were decoded and calibrated. This is the new URDF mesh / optical-flow extension, not a validation of the original Gaussian-splat pipeline. CPU results below are measured; Puget CUDA/RAFT timings remain to be collected.

## Optimization timing

| Episode prefix | Top: contour + motion | Left wrist | Right wrist |
|---|---:|---:|---:|
"""
for r in rows:
    text += f"| {r['episode'][8:16]} | {r['top_total_s']:.2f} s ({r['top_contour_s']:.2f} + {r['top_motion_s']:.2f}) | {r['left_wrist_s']:.2f} s | {r['right_wrist_s']:.2f} s |\n"
text += """
Top fits use 16 frames for contour ICP (30 iterations) and 24 adjacent frame pairs for motion refinement (350 Adam steps). Wrist fits use open-frame masks, a 180-candidate mount search, and up to 250 Nelder–Mead iterations. Timings exclude decoding, preparation, output rendering and environment setup. These are individual Mac CPU runs, not a controlled throughput benchmark.

## Wrist reliability

| Episode | Wrist | Training mask IoU | Held-out mask IoU | Status |
|---|---|---:|---:|---|
"""
for q in quality:
    for s, d in q["wrist"].items():
        text += f"| {q['episode'][8:16]} | {s} | {d['train_iou']:.3f} | {d['held_out_mask_iou']:.3f} | {d['status']} |\n"
text += """
Three of four wrist fits pass provisional heuristic checks. The second episode's right wrist is rejected and its candidate overlay is explicitly labeled. The first episode's left wrist has weaker mask stability and needs review. No ground-truth extrinsics were available: silhouette overlap and nearest-image-edge distance do not establish metric camera-pose accuracy. This sample is too small to estimate corpus-wide reliability.

## Videos and final overlays
"""
for q in quality:
    ep = q["episode"]
    text += f"""\n### {ep}

[Top optimization]({ep}/optimization.mp4) · [Top final overlay video]({ep}/final_overlay.mp4)

![Top final URDF overlays]({ep}/final_contact.jpg)

[Left wrist optimization]({ep}/wrist_left/optimization.mp4) · [Left wrist overlay]({ep}/wrist_left/final_overlay.mp4) · [Right wrist optimization]({ep}/wrist_right/optimization.mp4) · [Right wrist overlay]({ep}/wrist_right/final_overlay.mp4)

![Left wrist overlay]({ep}/wrist_left/final_overlay.jpg)

![Right wrist overlay]({ep}/wrist_right/final_overlay.jpg)
"""
text += """
## Decoding and scale

| Episode | Top decode + cache | Left wrist decode + cache | Right wrist decode + cache | Top render + validation |
|---|---:|---:|---:|---:|
"""
for r in rows:
    text += f"| {r['episode'][8:16]} | {r['top_decode_and_cache_s']:.2f} s | {r['left_wrist_decode_and_cache_s']:.2f} s | {r['right_wrist_decode_and_cache_s']:.2f} s | {r['top_render_and_validation_s']:.2f} s |\n"
text += """
Input episodes last approximately 100 and 117 seconds. Full top views were sampled at 5 Hz (960×600); wrists at 2 Hz (640×400). H.265 decoding and compressed cache writing dominate local wall time. The batch code supports bounded workers across GPUs 0,1,2, resumable outputs, explicit failures, optional cache deletion, and no videos by default. It has not been benchmarked on all ABC-130k. Reuse by station is only appropriate with verified camera/mount identities, which these MCAPs lack.

## What was delivered

- Timestamp-aligned ABC MCAP decoding, pinned I2RT and Menagerie models, tested YAM FK and correct jaw aperture mapping.
- A top-camera URDF contour / differentiable motion pipeline, with CPU/DIS validation and CUDA/RAFT options for Puget.
- A Cloak-inspired wrist pipeline with YAM initialization, camera-specific intrinsics, masks, explicit quality diagnostics and rejection.
- Actual optimization videos, full sampled-episode URDF overlays, transforms and separate timing records.
- Ubuntu/A6000 setup and run scripts, a bounded multi-GPU batch runner, numerical tests and a transfer bundle.

The original trained-Gaussian path, CUDA/RAFT execution, actual multi-GPU throughput, arbitrary-station initialization, other gripper revisions and full-dataset reliability remain unvalidated. See [Puget instructions](../../YAM_PUGET.md) for commands, conventions, references and limitations.
"""
(out / "REPORT.md").write_text(text)
print(json.dumps(rows, indent=2))
