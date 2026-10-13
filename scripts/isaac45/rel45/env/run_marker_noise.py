"""Run a script with the marker-image noise level set to MARKER_NOISE (upstream hard-codes 80), checkout unchanged.

ManiSkillSimulator.draw_markers (TacEx fem_based/mani_skill_sim.py) adds torch.rand_like noise * 80 inside every
marker patch. The released isaac45 demos have solid black dots (no noise). This wrapper replaces draw_markers with a
verbatim copy whose noise level comes from MARKER_NOISE, then runs the target with runpy. Chainable with the other
wrappers (run_grasp_thr.py, run_old_marker.py, run_with_render_overrides.py).
"""

import importlib.abc
import importlib.util
import os
import runpy
import sys

import torch

NOISE = float(os.environ["MARKER_NOISE"])
TARGET = "tacex.simulation_approaches.fem_based.mani_skill_sim"


def draw_markers(self, marker_uv: torch.Tensor) -> torch.Tensor:
    # verbatim upstream body (0dafa10 mani_skill_sim.py:303-368) except noise_level
    ManiSkillSimulator = type(self)
    device = "cuda:0"
    params = ManiSkillSimulator.patch_params
    circle_radius = params["circle_radius"]
    base_circle_radius = params["base_circle_radius"]
    super_resolution_ratio = params["super_resolution_ratio"]

    pad_size = 2 * circle_radius
    patch_size = 4 * circle_radius

    marker_image = torch.ones((self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]),
                              dtype=torch.float32, device=device) * 255
    marker_image = torch.nn.functional.pad(
        marker_image, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=255)

    uvs = marker_uv + 0.5 + pad_size
    u_floor = torch.floor(uvs[:, 0]).long()
    v_floor = torch.floor(uvs[:, 1]).long()
    u_frac = uvs[:, 0] - u_floor
    v_frac = uvs[:, 1] - v_floor

    patch_id_u = torch.floor(u_frac * super_resolution_ratio).long().clamp(0, super_resolution_ratio - 1)
    patch_id_v = torch.floor(v_frac * super_resolution_ratio).long().clamp(0, super_resolution_ratio - 1)
    marker_size = circle_radius
    patch_id_w = int((marker_size - base_circle_radius) * super_resolution_ratio)
    patch_id_w = max(0, min(patch_id_w, 49))

    patches = ManiSkillSimulator.patch_array[patch_id_u, patch_id_v, patch_id_w, :, :]

    for i in range(len(patches)):
        u_start = u_floor[i].item() - pad_size
        v_start = v_floor[i].item() - pad_size
        if marker_image.shape[1] - patch_size > u_start >= 0 and marker_image.shape[0] - patch_size > v_start >= 0:
            replace_idx = (
                slice(v_start, v_start + patch_size),
                slice(u_start, u_start + patch_size)
            )
            old_status = marker_image[replace_idx]
            new_status = torch.minimum(old_status, patches[i].float())
            marker_image[replace_idx] = new_status

    marker_image = marker_image[pad_size:-pad_size, pad_size:-pad_size]

    noise_level = NOISE
    if noise_level > 0:
        noise = torch.rand_like(marker_image) * noise_level
        marker_mask = marker_image < 255
        marker_image = torch.where(
            marker_mask,
            torch.clamp(marker_image + noise, 0, 255),
            marker_image)

    return marker_image


def patch(mod):
    mod.ManiSkillSimulator.draw_markers = draw_markers
    print(f"[marker noise] ManiSkillSimulator.draw_markers noise_level 80 -> {NOISE}", flush=True)


class PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
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
            patch(module)

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
