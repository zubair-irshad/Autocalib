"""Run decoding, mesh initialization, differentiable motion fitting, and videos."""

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from calibrate_abc_yam import fit as coarse_fit
from prepare_abc_yam import prepare
from refine_abc_yam_torch import fit as motion_fit
from render_abc_yam import render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/yam_puget"))
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--flow-backend", choices=["raft", "dis"], default="raft")
    p.add_argument("--wrist", action="store_true")
    p.add_argument("--summary-name", default="timing_summary")
    p.add_argument("--max-episodes", type=int, default=2)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--skip-videos", action="store_true")
    a = p.parse_args()
    paths = sorted(a.data.rglob("episode.mcap"))
    if a.max_episodes > 0:
        paths = paths[: a.max_episodes]
    if not paths:
        raise SystemExit(f"No episode.mcap files found under {a.data}")
    import torch

    if a.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is unavailable: install the CUDA PyTorch wheel and check nvidia-smi."
            )
        print("GPU:", torch.cuda.get_device_name(), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in paths:
        total = time.perf_counter()
        out = a.output / path.parent.name
        out.mkdir(exist_ok=True)
        if not (out / "preparation.json").exists():
            prepare(path, out)
        z = dict(np.load(out / "prepared.npz"))
        coarse = coarse_fit(z, out, a.iterations)
        del z
        refined = motion_fit(out, a.device, a.steps, a.flow_backend, a.pairs)
        if not a.skip_videos:
            render(out)
        wrist_results = {}
        if a.wrist:
            from calibrate_yam_wrist import fit as wrist_fit

            for side in ["left", "right"]:
                wrist_out = out / ("wrist_" + side)
                if not (wrist_out / "preparation.json").exists():
                    prepare(
                        path, wrist_out, stride=15, width=640, camera=side + "-wrist"
                    )
                wrist_results[side] = wrist_fit(
                    wrist_out, side, videos=not a.skip_videos
                )
        prep = json.load(open(out / "preparation.json"))
        row = {
            "episode": path.parent.name,
            "device": a.device,
            "flow_backend": a.flow_backend,
            "decode_seconds": prep["decode_seconds"],
            "coarse_fit_seconds": coarse["optimization_seconds"],
            "flow_preprocess_seconds": refined["preprocessing_seconds"],
            "motion_fit_seconds": refined["optimization_seconds"],
            "total_optimize_seconds": coarse["optimization_seconds"]
            + refined["optimization_seconds"],
            "this_run_wall_seconds": time.perf_counter() - total,
            "output": str(out.resolve()),
        }
        row["wrist_optimization_seconds"] = sum(
            r.get("optimization_seconds", 0) for r in wrist_results.values()
        )
        summary.append(row)
        print(json.dumps(row, indent=2), flush=True)
        (a.output / (a.summary_name + ".json")).write_text(
            json.dumps({"platform": platform.platform(), "results": summary}, indent=2)
        )
    import csv

    with (a.output / (a.summary_name + ".csv")).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
