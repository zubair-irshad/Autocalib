"""Decode ABC MCAP video and nearest-time synchronized measured joint states."""

import argparse
import json
import time
from pathlib import Path

import av
import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def stamp(d):
    return d.timestamp.seconds * 10**9 + d.timestamp.nanos


def prepare(path, out, stride=6, width=960, camera="top-left"):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    streams = {}
    info = {}
    times = []
    frames = []
    codec = None
    count = 0
    with path.open("rb") as f:
        summary = make_reader(f).get_summary()
        available = {c.topic for c in summary.channels.values()}
    if (
        camera == "top-left"
        and "/top-left-camera" not in available
        and "/top-camera" in available
    ):
        camera = "top"
    if f"/{camera}-camera" not in available:
        raise ValueError(
            f"Missing camera stream: {camera}; available: {sorted(available)}"
        )
    topics = [f"/{camera}-camera", f"/{camera}-camera-info"] + [
        f"/{s}-{a}-state" for s in ["left", "right"] for a in ["arm", "ee"]
    ]
    with path.open("rb") as f:
        for _, ch, msg, d in make_reader(
            f, decoder_factories=[DecoderFactory()]
        ).iter_decoded_messages(topics=topics):
            t = stamp(d)
            if ch.topic.endswith("-info"):
                info = {
                    "K": list(d.K),
                    "D": list(d.D),
                    "width": d.width,
                    "height": d.height,
                }
            elif ch.topic == f"/{camera}-camera":
                if codec is None:
                    codec = av.CodecContext.create(
                        "hevc" if d.format == "h265" else "h264", "r"
                    )
                packet = av.Packet(d.data)
                packet.pts = t
                packet.dts = t
                for frame in codec.decode(packet):
                    if count % stride == 0:
                        im = frame.to_ndarray(format="bgr24")
                        h, w = im.shape[:2]
                        frames.append(cv2.resize(im, (width, round(h * width / w))))
                        times.append(frame.pts if frame.pts is not None else t)
                    count += 1
            else:
                streams.setdefault(ch.topic, []).append([t, *d.position])
        for frame in codec.decode(None):
            if count % stride == 0:
                im = frame.to_ndarray(format="bgr24")
                h, w = im.shape[:2]
                frames.append(cv2.resize(im, (width, round(h * width / w))))
                times.append(frame.pts)
            count += 1
    ts = np.array(times, dtype=np.int64)
    values = {}
    maxdt = {}
    for key, rows in streams.items():
        # Keep nanosecond clocks as integers (float timestamps lose precision).
        rt = np.array([r[0] for r in rows], dtype=np.int64)
        v = np.array([r[1:] for r in rows])
        order = np.argsort(rt)
        rt = rt[order]
        v = v[order]
        i = np.searchsorted(rt, ts).clip(0, len(rt) - 1)
        j = (i - 1).clip(0, len(rt) - 1)
        i = np.where(np.abs(rt[j] - ts) < np.abs(rt[i] - ts), j, i)
        name = key.strip("/").replace("-", "_")
        values[name] = v[i]
        maxdt[name] = float(np.max(np.abs(rt[i] - ts)) / 1e6)
    K = np.array(info["K"]).reshape(3, 3)
    K[:2] *= width / info["width"]
    distortion = np.asarray(info["D"])
    if np.any(np.abs(distortion) > 1e-9):
        frames = [cv2.undistort(im, K, distortion) for im in frames]
    np.savez_compressed(
        out / "prepared.partial.npz",
        frames=np.array(frames),
        timestamps_ns=ts,
        K=K,
        D=np.zeros_like(distortion),
        source_distortion=distortion,
        **values,
    )
    (out / "prepared.partial.npz").replace(out / "prepared.npz")
    meta = {
        "episode": path.parent.name,
        "camera": f"/{camera}-camera",
        "source_frames": count,
        "sampled_frames": len(frames),
        "stride": stride,
        "resolution": [width, frames[0].shape[0]],
        "seconds": float((ts[-1] - ts[0]) / 1e9),
        "decode_seconds": time.perf_counter() - start,
        "nearest_joint_max_delta_ms": maxdt,
    }
    (out / "preparation.json").write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)
    inds = np.linspace(0, len(frames) - 1, 8, dtype=int)
    thumb = []
    for i in inds:
        im = cv2.resize(frames[i], (480, 300))
        cv2.putText(
            im, f"{i}: {(ts[i] - ts[0]) / 1e9:.1f}s", (10, 25), 0, 0.7, (0, 255, 255), 2
        )
        thumb.append(im)
    cv2.imwrite(
        str(out / "contact.jpg"),
        np.concatenate(
            [np.concatenate(thumb[:4], axis=1), np.concatenate(thumb[4:], axis=1)],
            axis=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("ABC-130k"))
    p.add_argument("--output", type=Path, default=Path("outputs/yam"))
    a = p.parse_args()
    for path in sorted(a.data.rglob("episode.mcap")):
        if not (a.output / path.parent.name / "preparation.json").exists():
            prepare(path, a.output / path.parent.name)
