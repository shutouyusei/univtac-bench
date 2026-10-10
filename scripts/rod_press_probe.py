"""Read-out of the rod pressing probe (envs/rod_press.py), fixed before the probe is run.

Usage: python scripts/rod_press_probe.py data/rod_press/metadata.json [--seed 0] [--out dir] [--snr 3.0]

Per board offset, over the settled steps: mean and std of the marker-field RMS change, the vertical shear
(mean marker displacement), the net vertical contact force per gel, the press depth and the in-hand slip.
Noise is the std of the same quantities at offset 0. The verdict has two parts, both needed to go on to a
task: (1) at |offset| = 1 mm the marker RMS change exceeds ``snr`` times the offset-0 noise; (2) the vertical
force change has opposite signs for +1 mm and -1 mm (the reading carries the direction of the motion).
Figures: readings against offset, and the full time series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GELS = ('left_tactile', 'right_tactile')


def summarize(records: list[dict]) -> dict[float, dict[str, float]]:
    """Settled-step statistics per target offset (metres), in the order the offsets were visited."""
    out: dict[float, dict[str, float]] = {}
    for target in dict.fromkeys(r['target'] for r in records):
        rows = [r for r in records if r['target'] == target and r['settled']]
        if not rows:
            continue
        rms = np.array([r['marker_rms_px'] for r in rows])
        shear = np.array([r['marker_mean_px'][1] for r in rows])
        fz = np.array([sum(r[f'{g}_force_N'][2] for g in GELS if f'{g}_force_N' in r) for r in rows])
        depth = np.array([max(r.get(f'{g}_press_depth_mm', 0.0) for g in GELS) for r in rows])
        slip = np.array([r['inhand_slip_m'] for r in rows])
        out[target] = {
            'n': len(rows),
            'rms_mean': float(rms.mean()), 'rms_std': float(rms.std()),
            'shear_mean': float(shear.mean()), 'shear_std': float(shear.std()),
            'fz_mean': float(fz.mean()), 'fz_std': float(fz.std()),
            'depth_mean': float(depth.mean()),
            'slip_mean': float(slip.mean()),
        }
    return out


def verdict(summary: dict[float, dict[str, float]], snr: float = 3.0) -> dict:
    """The two pre-registered checks at |offset| = 1 mm against the offset-0 noise."""
    zero = summary.get(0.0)
    up, down = summary.get(0.001), summary.get(-0.001)
    if zero is None or up is None or down is None:
        return {'ok': False, 'reason': 'offsets 0, +1 mm and -1 mm are all needed'}
    noise = max(zero['rms_std'], 1e-9)
    snr_up = (up['rms_mean'] - zero['rms_mean']) / noise
    snr_down = (down['rms_mean'] - zero['rms_mean']) / noise
    visible = min(snr_up, snr_down) > snr
    d_up = up['fz_mean'] - zero['fz_mean']
    d_down = down['fz_mean'] - zero['fz_mean']
    directional = d_up * d_down < 0 and min(abs(d_up), abs(d_down)) > zero['fz_std']
    return {
        'ok': bool(visible and directional),
        'rms_snr_up': float(snr_up), 'rms_snr_down': float(snr_down),
        'fz_delta_up': float(d_up), 'fz_delta_down': float(d_down), 'fz_noise': float(zero['fz_std']),
        'visible': bool(visible), 'directional': bool(directional),
    }


def plot(records: list[dict], summary: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    offsets = np.array(list(summary)) * 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(axes, ('rms', 'shear', 'fz'),
                              ('marker RMS change [px]', 'vertical shear [px]', 'net vertical force [N]')):
        m = [summary[o][f'{key}_mean'] for o in summary]
        s = [summary[o][f'{key}_std'] for o in summary]
        ax.errorbar(offsets, m, yerr=s, fmt='o-')
        ax.set_xlabel('board offset [mm]')
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out / 'probe_vs_offset.png', dpi=150)

    steps = [r['step'] for r in records]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(steps, [r['offset'] * 1000 for r in records]); axes[0].set_ylabel('offset [mm]')
    axes[1].plot(steps, [r['marker_rms_px'] for r in records]); axes[1].set_ylabel('marker RMS [px]')
    for g in GELS:
        axes[2].plot(steps, [r[f'{g}_force_N'][2] for r in records], label=g)
    axes[2].set_ylabel('Fz [N]'); axes[2].legend()
    axes[3].plot(steps, [r['inhand_slip_m'] * 1000 for r in records]); axes[3].set_ylabel('in-hand slip [mm]')
    axes[3].set_xlabel('control step')
    fig.tight_layout()
    fig.savefig(out / 'probe_timeseries.png', dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('metadata', type=Path)
    ap.add_argument('--seed', default='0')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--snr', type=float, default=3.0)
    args = ap.parse_args()

    records = json.load(open(args.metadata))[args.seed]['probe']
    summary = summarize(records)
    print(f"{'offset[mm]':>10} {'n':>3} {'rms[px]':>14} {'shear[px]':>14} {'Fz[N]':>16} {'depth[mm]':>9} {'slip[mm]':>8}")
    for o, s in summary.items():
        print(f"{o * 1000:>10.1f} {s['n']:>3} {s['rms_mean']:>7.3f}±{s['rms_std']:<6.3f} "
              f"{s['shear_mean']:>7.3f}±{s['shear_std']:<6.3f} {s['fz_mean']:>8.4f}±{s['fz_std']:<7.4f} "
              f"{s['depth_mean']:>9.3f} {s['slip_mean'] * 1000:>8.2f}")
    v = verdict(summary, args.snr)
    print('verdict:', json.dumps(v, indent=1))
    out = args.out or args.metadata.parent
    out.mkdir(parents=True, exist_ok=True)
    plot(records, summary, out)
    json.dump({'summary': {str(k): s for k, s in summary.items()}, 'verdict': v},
              open(out / 'probe_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
