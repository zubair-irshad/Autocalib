# YAM / ABC-130k calibration on Puget

This is an **additive mesh-based calibration extension**. It does not run or validate the repository's original trained-Gaussian / AllTracker calibration pipeline. The existing scripts assume other robot checkpoints and contain machine-specific paths. No trained YAM Gaussian checkpoint was available.

The new path uses actual I2RT URDF meshes, known camera intrinsics and joint states, contour initialization, and differentiable optical-flow fitting. It needs no robot appearance training. Wrist calibration adapts Cloak's silhouette approach to ABC's YAM grippers. The entire path was exercised on the two supplied episodes on Apple Silicon CPU; CUDA, RAFT, Ubuntu EGL and multi-GPU execution still need validation on Puget.

## Puget: Ubuntu, 3 × RTX A6000, driver 580.65.06

Copy this repository or extract `yam_puget_bundle.tar.gz` onto Puget. The bundle includes code, pinned robot meshes and licenses, but excludes episodes, environment files, and generated videos. If the robot meshes were omitted by a Git transfer, the setup downloads the pinned assets automatically. Place the supplied episodes under a directory containing `episode_*/episode.mcap`, at any nesting depth.

```bash
# From the extracted bundle/repository root:
bash scripts/setup_yam_puget.sh

# First run: one GPU, both supplied episodes, top and both wrist cameras, all videos.
bash scripts/run_yam_puget.sh /absolute/path/ABC-130k --wrist
```

The setup creates `.venv-yam-puget` and installs PyTorch 2.5.1 / torchvision 0.20.1 with CUDA 12.1 wheels. These match the local PyTorch version; CUDA 13.0 in nvidia-smi is the driver's advertised capability, not a requirement to install a CUDA 13 PyTorch wheel. Python 3.11 is required; the setup uses `uv` to provision it if available, otherwise `python3.11 -m venv`. Install `ffmpeg` if missing. Ubuntu also needs functioning NVIDIA EGL libraries for headless MuJoCo rendering. System CUDA toolkit compilation is unnecessary for these scripts.

The first command uses pretrained torchvision RAFT and downloads its weights. **RAFT is optional and untested here.** Use the tested DIS flow backend for a first parity/speed run:

```bash
bash scripts/run_yam_puget.sh /absolute/path/ABC-130k --wrist --flow-backend dis --output outputs/yam_puget_dis
```

For existing decoded caches, run stages directly:

```bash
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
.venv-yam-puget/bin/python scripts/calibrate_abc_yam.py --episode outputs/yam_puget/episode_UUID
.venv-yam-puget/bin/python scripts/refine_abc_yam_torch.py --episode outputs/yam_puget/episode_UUID --device cuda --flow-backend raft
.venv-yam-puget/bin/python scripts/render_abc_yam.py --episode outputs/yam_puget/episode_UUID
.venv-yam-puget/bin/python scripts/calibrate_yam_wrist.py --episode outputs/yam_puget/episode_UUID
```

## Parallel processing

```bash
# Start with a pilot, not the entire corpus. One episode subprocess per physical GPU.
.venv-yam-puget/bin/python scripts/batch_yam.py \
  --data /absolute/path/ABC-130k \
  --output outputs/yam_batch_pilot \
  --gpus 0,1,2 --max-episodes 100 --wrist --flow-backend dis --drop-decoded
```

The batch runner bounds concurrency, gives each process its own CUDA/EGL device, writes per-episode logs, continues after failures, records a JSONL manifest, and resumes completed episodes. It skips videos by default; add `--videos` for an audit subset. `--drop-decoded` deletes only generated decode caches after successful completion; it never deletes source MCAPs. A completed job can contain a rejected wrist fit: completion and calibration quality are deliberately separate. `--force` reruns completed jobs. `--dry-run` prints commands without launching jobs.

Optimize and benchmark a representative pilot before removing `--max-episodes`. Two scenes do not establish corpus-wide accuracy or throughput. Decoding/compressing the cache currently dominates local wall time. For corpus scale, the next useful engineering steps are one-pass multi-camera decoding, persistent workers retaining RAFT/model state, compact calibration-frame caches, and batched silhouette rendering. Those improvements are **not implemented**. The current batch runner is parallel and bounded, but creates a new process/model per episode. Avoid assuming a threefold speedup: CPU decode and storage may become shared bottlenecks.

Verified station/camera-mount IDs would allow calibration reuse, followed by per-episode checks. These two MCAPs do not contain such an ID. Do not infer shared calibration from task/operator or matching focal lengths alone.

## Outputs and coordinate conventions

Each episode produces:

- `calibration_coarse.json`: top-camera contour fit and recorded iterates.
- `calibration.json`: final top transforms, PyTorch motion fit, synchronized optimization timing, flow backend and device/version metadata.
- `optimization.mp4`: actual top solver states, including contour and motion stages.
- `final_overlay.mp4`: full sampled episode, recorded view next to rendered URDF overlay.
- `validation.json`: held-out nearest-edge proxy metrics. These are **not ground-truth pose errors**.
- `wrist_left/` and `wrist_right/`: synchronized data, target-mask diagnostics, extrinsics, optimization/overlay videos, acceptance status and rejection reasons.
- Run-level timing JSON/CSV; the batch runner adds `batch_results.jsonl`, per-episode `run.log`, and `complete.json`.

Top pose vectors are `[rx, ry, rz, tx, ty, tz]`, with axis-angle rotation in radians and translation in metres: `p_top_camera = R @ p_arm_base + t`. Both arm transforms share the top camera frame.

Wrist vectors use the same order but map **camera coordinates into the I2RT v1 URDF `gripper` frame**. `T_gripper_camera` is stored explicitly. For a measured configuration:

```
T_base_camera(q) = T_base_gripper(q) @ T_gripper_camera
```

The camera convention is OpenCV: x right, y down, z forward. This is not Cloak's `[translation, XYZ Euler]` vector convention; do not interchange them. Recorded focal lengths and principal points are retained; the wrist renderer handles unequal fx/fy. The linked episodes have zero recorded distortion. Nonzero standard distortion coefficients are undistorted before fitting; fisheye schemas would need a separate adapter.

ABC aperture is **0=closed, 1=open**. In the vendored I2RT v1 URDF, both finger slide coordinates are `-0.04695 * aperture`. This was checked against the mesh geometry and is covered by a test.

## Wrist method and reliability

Cloak's published calibration entry point targets DROID/Franka. Its repository also contains YAM rendering assets and a YAM mount prior, but those are not evidence of ABC-130k calibration performance. This adaptation uses the low-intensity / low-temporal-variance idea with:

- ABC MCAP timestamp matching and aperture >= 0.95 for open frames;
- actual per-camera intrinsics and I2RT YAM geometry;
- both bottom corners in the ROI, rather than DROID's asymmetric ROI;
- a bounded mount search before local Nelder–Mead silhouette fitting;
- separate open-frame subsets to diagnose mask consistency;
- explicit low-IoU/unstable-mask rejection, rather than dropping a fixed percentage.

`accepted_for_review` means heuristic checks passed, not that extrinsics are verified. The provisional thresholds (train IoU >= 0.60, held-out mask IoU >= 0.55, mask agreement >= 0.55) were not validated on a large dataset. Two small visible finger silhouettes can leave pose ambiguities. Held-out masks at the same opening are not independent geometric ground truth. Grasped objects, shadows, limited open frames, weak motion and different gripper revisions can cause failures. Inspect closed/intermediate-aperture overlays too. The second supplied episode's right wrist failed the checks and is retained as a failure example.

The top-camera prior is a manually chosen **generic tabletop pose prior**, shared across both supplied scenes; there are no manually annotated image keypoints. It is not a global initialization algorithm for arbitrary station layouts. Joint states and intrinsics are held fixed. Camera mounts, cables and grasped objects are not part of the rendered URDF, and overlays do not model occlusion by real objects. The wrist views render the camera arm's gripper, not the complete scene or the other arm.

## Validation and sources

```bash
.venv-yam-puget/bin/python -m unittest discover -s tests -p 'test_yam*.py' -v
MUJOCO_GL=egl .venv-yam-puget/bin/python scripts/verify_yam_projection.py
```

Local checks covered URDF/MuJoCo FK agreement, aperture direction, rotation gradients at zero, projection against OpenCV, the complete CPU runner, and batch command generation. The rendered anisotropic-camera projection check had <0.70 px maximum error. CUDA execution and actual parallel scheduling were not tested locally.

- [ABC-130k data format](https://huggingface.co/datasets/XDOF/ABC-130k)
- [I2RT URDF at the pinned revision](https://github.com/i2rt-robotics/i2rt/tree/5b72c47239bd056d0fa6c1a39edeb0537c89443c/i2rt/robot_models/arm/yam/v1)
- [MuJoCo Menagerie YAM](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/i2rt_yam)
- [Cloak wrist algorithm](https://github.com/Stanford-TML/cloak/blob/174aa4e8eeec2141e7b5e02978dba9b1c9068bbf/preprocessing/preprocess_wrist_extrinsics.py)
- [Cloak YAM prior](https://github.com/Stanford-TML/cloak/blob/174aa4e8eeec2141e7b5e02978dba9b1c9068bbf/src/openpi/constants.py)
- [Official PyTorch 2.5.1 CUDA wheel commands](https://pytorch.org/get-started/previous-versions/)

Robot downloads are pinned and recorded with SHA-256 hashes in `third_party/yam/*/provenance.json`. Upstream MIT licenses are retained. The current I2RT v1 URDF is used for fitting/rendering; the older I2RT model and Menagerie description are included as references and are not mixed into its kinematics.
