"""Check MuJoCo rendered pixels against non-centred, anisotropic camera intrinsics."""

import mujoco
import numpy as np
from calibrate_abc_yam import ROOT, seed
from yam_renderer import YamRenderer

K = np.array([[224.0, 0, 311.0], [0, 240.0, 195.0], [0, 0, 1.0]])
r = YamRenderer(ROOT / "third_party/yam/i2rt/yam.urdf", K, 640, 400)
errors = []
for p in [
    np.array([0.0, 0.0, 1.0]),
    np.array([0.3, 0.2, 1.0]),
    np.array([-0.3, -0.2, 1.0]),
]:
    r.setup(np.zeros(8), seed("left"))
    r.renderer.scene.ngeom = 1
    g = r.renderer.scene.geoms[0]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, 0.015),
        p,
        np.eye(3).ravel(),
        np.array([1.0, 0.0, 0.0, 1.0]),
    )
    im = r.resample(r.renderer.render())
    mask = (im[:, :, 0] > 30) & (im[:, :, 0] > im[:, :, 1] * 2)
    y, x = np.nonzero(mask)
    actual = np.array([x.mean(), y.mean()])
    expected = (K @ p)[:2] / p[2]
    error = float(np.linalg.norm(actual - expected))
    errors.append(error)
    print("expected", expected, "rendered", actual, "error_px", error)
    assert error < 1.5
r.close()
print("max_error_px", max(errors))
