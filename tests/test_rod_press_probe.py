"""The rod-press probe read-out: settled-only statistics and the two pre-registered checks."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'rod_press_probe', Path(__file__).resolve().parents[1] / 'scripts' / 'rod_press_probe.py')
probe = importlib.util.module_from_spec(spec)
sys.modules['rod_press_probe'] = probe
spec.loader.exec_module(probe)


def _rec(target, rms, fz, settled=True, step=0):
    return {
        'step': step, 'target': target, 'offset': target, 'settled': settled,
        'marker_rms_px': rms, 'marker_mean_px': [0.0, rms], 'inhand_slip_m': 0.0,
        'left_tactile_force_N': [0.0, 0.0, fz / 2], 'right_tactile_force_N': [0.0, 0.0, fz / 2],
        'left_tactile_press_depth_mm': 0.3, 'right_tactile_press_depth_mm': 0.3,
    }


def _records(up_rms, down_rms, up_fz, down_fz):
    rows = []
    for target, rms, fz in ((0.0, 1.0, -0.5), (0.001, up_rms, up_fz), (-0.001, down_rms, down_fz)):
        rows.append(_rec(target, 99.0, 99.0, settled=False))  # ramp steps must not count
        for k in range(4):
            rows.append(_rec(target, rms + 0.1 * (k % 2), fz + 0.01 * (k % 2), step=k))
    return rows


def test_summarize_uses_only_settled_steps_in_visit_order():
    s = probe.summarize(_records(3.0, 3.0, -1.0, 0.0))
    assert list(s) == [0.0, 0.001, -0.001]
    assert s[0.0]['n'] == 4
    assert abs(s[0.0]['rms_mean'] - 1.05) < 1e-9
    assert s[0.001]['rms_mean'] < 10  # the 99 ramp rows are excluded


def test_verdict_passes_when_visible_and_directional():
    v = probe.verdict(probe.summarize(_records(3.0, 3.0, -1.0, 0.0)))
    assert v['visible'] and v['directional'] and v['ok']


def test_verdict_fails_when_change_is_inside_noise():
    v = probe.verdict(probe.summarize(_records(1.05, 1.05, -1.0, 0.0)))
    assert not v['visible'] and not v['ok']


def test_verdict_fails_when_force_has_the_same_sign_both_ways():
    v = probe.verdict(probe.summarize(_records(3.0, 3.0, -1.0, -1.0)))
    assert v['visible'] and not v['directional'] and not v['ok']


def test_verdict_needs_all_three_offsets():
    rows = [r for r in _records(3.0, 3.0, -1.0, 0.0) if r['target'] != -0.001]
    assert probe.verdict(probe.summarize(rows))['ok'] is False
