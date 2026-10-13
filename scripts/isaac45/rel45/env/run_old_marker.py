"""Run a script with the gsmini marker code of the authors' eval commit (a9abc0d), the rest of the checkout unchanged.

Between a9abc0d and main 0dafa10 two gsmini-relevant marker changes landed (TacEx 35bc438 and envs/sensors/tactile.py):
  1. VisionTactileSensorUIPC._gen_marker_weight: nearest-face barycentric weights on the raw camera-frame surface
     points (a9abc0d) -> Delaunay on recentred surface points (main).
  2. VisualTactileSensor.setup(): a9abc0d re-captured the reference vertices with marker_motion_sim.init_vertices()
     after the attachment was set up; main captures them only in the simulator constructor.
This wrapper restores both by patching the two modules right after they are imported, then runs the target with
runpy (e.g. scripts/isaac45/run_with_render_overrides.py scripts/collect_data.py ...). Prints a line per patch.
"""

import importlib.abc
import importlib.util
import runpy
import os
import sys

import numpy as np

MARKER_MOD = "tacex.simulation_approaches.fem_based.sim.tactile_sensor_sapienipc_modified"
TACTILE_MOD = "envs.sensors.tactile"


def old_gen_marker_weight(self, marker_pts):
    # verbatim body of a9abc0d (third_party/TacEx/.../tactile_sensor_sapienipc_modified.py)
    from sklearn.neighbors import NearestNeighbors

    surface_pts = self.init_surface_vertices_camera.cpu().numpy()[:, :2]

    face_idx_to_surface_idx = np.zeros(np.max(self.faces_on_surfaces)+1, dtype=np.int32)
    face_idx_to_surface_idx[self.vertices_on_surface] = np.arange(surface_pts.shape[0])
    faces_v_on_surface = surface_pts[face_idx_to_surface_idx[self.faces_on_surfaces.flatten()]].reshape(-1, 3, 2)
    f_centers = np.mean(faces_v_on_surface, axis=1)

    nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(f_centers)
    distances, face_idx = nbrs.kneighbors(marker_pts)

    marker_pts_surface_idx = []
    marker_pts_surface_weight = []
    valid_marker_idx = []

    for i in range(marker_pts.shape[0]):
        possible_face_ids = face_idx[i]
        p = marker_pts[i]
        for possible_face_id in possible_face_ids.tolist():
            face_vertices_idx = face_idx_to_surface_idx[self.faces_on_surfaces[possible_face_id]]
            vertices_of_face_i = surface_pts[face_vertices_idx]
            p0, p1, p2 = vertices_of_face_i
            A = np.stack([p1 - p0, p2 - p0], axis=1)
            w12 = np.linalg.inv(A) @ (p - p0)
            if possible_face_id == possible_face_ids[0]:
                marker_pts_surface_idx.append(face_vertices_idx)
                marker_pts_surface_weight.append(np.array([1 - w12.sum(), w12[0], w12[1]]))
                valid_marker_idx.append(i)
                if w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                    break
            elif w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                marker_pts_surface_idx[-1] = face_vertices_idx
                marker_pts_surface_weight[-1] = np.array([1 - w12.sum(), w12[0], w12[1]])
                valid_marker_idx[-1] = i
                break

    valid_marker_idx = np.array(valid_marker_idx).astype(np.int32)
    marker_pts = marker_pts[valid_marker_idx]
    marker_pts_surface_idx = np.stack(marker_pts_surface_idx)
    marker_pts_surface_weight = np.stack(marker_pts_surface_weight)
    assert np.allclose(
        (surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]).sum(1)[:, :2],
        marker_pts,
    ), (
        "max err:"
        f" {np.abs((surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]).sum(1)[:, :2] - marker_pts).max()}"
    )

    return marker_pts_surface_idx, marker_pts_surface_weight


def timed(name, fn):
    import time

    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"[old marker] {name} took {time.perf_counter() - t:.3f} s", flush=True)
        return out

    return wrapper


def patch_marker(mod):
    mod.VisionTactileSensorUIPC._gen_marker_weight = timed("_gen_marker_weight", old_gen_marker_weight)
    mod.VisionTactileSensorUIPC.init_vertices = timed("init_vertices", mod.VisionTactileSensorUIPC.init_vertices)
    print(f"[old marker] {MARKER_MOD}.VisionTactileSensorUIPC._gen_marker_weight <- a9abc0d", flush=True)


def patch_tactile(mod):
    original = mod.VisualTactileSensor.setup

    def setup(self):
        original(self)
        self.sensor.marker_motion_simulator.marker_motion_sim.init_vertices()
        print("[old marker] setup: init_vertices re-called", flush=True)

    mod.VisualTactileSensor.setup = setup
    print(f"[old marker] {TACTILE_MOD}.VisualTactileSensor.setup re-calls init_vertices() (a9abc0d)", flush=True)


# OLD_MARKER_SETUP=0 keeps main's setup() (reference vertices captured only in the constructor) and restores only the
# a9abc0d marker weights (NN-barycentric on raw camera-frame points, extrapolating outside the surface hull).
PATCHES = {MARKER_MOD: patch_marker}
if os.environ.get("OLD_MARKER_SETUP", "1") != "0":
    PATCHES[TACTILE_MOD] = patch_tactile


class PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            PATCHES[fullname](module)

        spec.loader.exec_module = exec_and_patch
        return spec


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.meta_path.insert(0, PatchFinder())
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
