"""Resume-safe, bounded parallel ABC calibration: one subprocess per GPU slot."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/yam_batch")
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--flow-backend", choices=["dis", "raft"], default="dis")
    p.add_argument("--wrist", action="store_true")
    p.add_argument("--videos", action="store_true")
    p.add_argument("--drop-decoded", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    paths = sorted(a.data.resolve().rglob("episode.mcap"))
    paths = paths[: a.max_episodes] if a.max_episodes else paths
    if not paths:
        raise SystemExit("No episodes found")
    ids = [p.parent.name for p in paths]
    if len(ids) != len(set(ids)):
        raise SystemExit(
            "Duplicate episode IDs found; separate roots or rename output identifiers before running."
        )
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in a.gpus.split(",")]
    lock = threading.Lock()
    jobs = iter(paths)
    results = []

    def worker(gpu):
        while True:
            with lock:
                try:
                    ep = next(jobs)
                except StopIteration:
                    return
            out = a.output / ep.parent.name
            done = out / "complete.json"
            if done.exists() and not a.force:
                continue
            out.mkdir(exist_ok=True)
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=str(gpu),
                MUJOCO_EGL_DEVICE_ID=str(gpu),
                MUJOCO_GL="egl",
                OMP_NUM_THREADS="4",
            )
            cmd = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_yam_pipeline.py"),
                "--data",
                str(ep.parent),
                "--output",
                str(a.output),
                "--max-episodes",
                "1",
                "--device",
                "cuda",
                "--flow-backend",
                a.flow_backend,
                "--summary-name",
                ep.parent.name,
            ]
            if a.wrist:
                cmd += ["--wrist"]
            if not a.videos:
                cmd += ["--skip-videos"]
            if a.dry_run:
                with lock:
                    print(json.dumps({"gpu": gpu, "command": cmd}), flush=True)
                continue
            start = time.perf_counter()
            with (out / "run.log").open("w") as log:
                r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            row = {
                "episode": ep.parent.name,
                "gpu": gpu,
                "returncode": r.returncode,
                "wall_seconds": time.perf_counter() - start,
                "log": str(out / "run.log"),
            }
            if r.returncode == 0:
                cal = json.load(open(out / "calibration.json"))
                row["top_optimization_seconds"] = cal["optimization_seconds"] + cal.get(
                    "coarse_optimization_seconds", 0
                )
                if a.wrist:
                    row["wrist_statuses"] = {
                        s: json.load(
                            open(out / ("wrist_" + s) / "wrist_calibration.json")
                        )["status"]
                        for s in ["left", "right"]
                    }
                done.write_text(json.dumps(row, indent=2))
                if a.drop_decoded:
                    # Delete only generated caches inside this completed episode, never input MCAPs.
                    for folder in [out, out / "wrist_left", out / "wrist_right"]:
                        for name in ["prepared.npz", "preparation.json"]:
                            (folder / name).unlink(missing_ok=True)
            with lock:
                results.append(row)
                with (a.output / "batch_results.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(worker, gpus))
    if any(r["returncode"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
