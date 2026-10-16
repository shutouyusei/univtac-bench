"""Run an upstream script with RTX settings applied after Isaac Lab's rendering preset, which would
overwrite the same keys given as --kit_args.

    UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false" \\
        python scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py insert_hole clean ACT/deploy ...
"""

import json
import os
import runpy
import sys


def parse(spec: str) -> list[tuple[str, object]]:
    """``key=value;...`` -> pairs; a value is JSON (false, 2, 0.5) or else the string itself."""
    pairs = []
    for item in filter(None, spec.split(";")):
        key, value = item.split("=", 1)
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value.strip()
        pairs.append((key.strip(), parsed))
    return pairs


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    overrides = parse(os.environ.get("UNIVTAC45_RENDER_OVERRIDES", ""))

    from isaaclab.app import app_launcher

    original = app_launcher.AppLauncher._set_rendering_mode_settings

    def with_overrides(self, launcher_args):
        original(self, launcher_args)
        import carb
        from isaacsim.core.utils.carb import set_carb_setting

        settings = carb.settings.get_settings()
        for key, value in overrides:
            set_carb_setting(settings, key, value)
            print(f"[render override] {key} = {value!r} (now {settings.get(key)!r})", flush=True)

    app_launcher.AppLauncher._set_rendering_mode_settings = with_overrides
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
