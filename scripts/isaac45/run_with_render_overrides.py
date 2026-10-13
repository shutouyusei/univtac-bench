"""Run an upstream script (e.g. scripts/eval_policy.py) with RTX settings overridden after Isaac Lab's rendering
preset is applied.

Isaac Lab's AppLauncher applies the rendering-mode preset (apps/rendering_modes/<mode>.kit) as carb settings after
Kit has started, so `--kit_args "--/rtx/..."` cannot change a key the preset sets. This wrapper applies
UNIVTAC45_RENDER_OVERRIDES right after the preset and then runs the target script unchanged, so the simulator code
stays upstream and the overrides are explicit in the command line.

    UNIVTAC45_RENDER_OVERRIDES="/rtx/directLighting/sampledLighting/enabled=false;/rtx-transient/dldenoiser/enabled=false" \
        python scripts/isaac45/run_with_render_overrides.py scripts/eval_policy.py insert_hole clean ACT/deploy ...

Values: true / false, integers, floats, otherwise strings. Each applied setting is printed.
"""

import os
import runpy
import sys


def parse(spec: str) -> list[tuple[str, object]]:
    pairs = []
    for item in filter(None, (s.strip() for s in spec.split(";"))):
        key, value = item.split("=", 1)
        low = value.strip().lower()
        if low in ("true", "false"):
            parsed: object = low == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
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
