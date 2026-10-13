"""Run a script with the adaptive-grasp depth threshold set to GRASP_THR (mm) for every task, checkout unchanged.

BaseTask.load_robot_and_sensors resolves cfg.adaptive_grasp_depth_threshold (robot default 27.5 mm for gsmini,
used by atom.close_gripper in pre_move). This wrapper overrides the resolved value right after that method, then
runs the target with runpy (e.g. scripts/isaac45/run_with_render_overrides.py scripts/collect_data.py ...).
"""

import importlib.abc
import importlib.util
import os
import runpy
import sys

THR = float(os.environ["GRASP_THR"])
TARGET = "envs._base_task"


def patch(mod):
    original = mod.BaseTask.load_robot_and_sensors

    def load_robot_and_sensors(self, cfg):
        cfg = original(self, cfg)
        print(f"[grasp thr] adaptive_grasp_depth_threshold {cfg.adaptive_grasp_depth_threshold} -> {THR}", flush=True)
        cfg.adaptive_grasp_depth_threshold = THR
        return cfg

    mod.BaseTask.load_robot_and_sensors = load_robot_and_sensors


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
