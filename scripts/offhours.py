"""Run one queued unit of work (one training or one evaluation) only outside research hours.

    tsp -L act_lift_bottle_train ~/miniforge3/envs/UniVTAC/bin/python scripts/offhours.py \
        --kind act_train --estimate-min 40 -- bash policy/ACT/baseline.sh lift_bottle train

Before starting, the unit waits until
  * the clock is inside the off-hours window (``--config``, default ``~/.config/univtac-offhours.yaml``;
    weekdays 21:00-08:00, weekends and listed holidays all day),
  * the pause file (``--pause-file``, default ``~/.offhours_pause``) is absent,
  * other compute processes hold at most ``gpu_busy_mib`` of GPU memory (a resident viewer does not block),
  * the estimated duration plus ``margin_min`` ends before the window closes. The estimate is the larger of
    ``--estimate-min`` and 1.1 x the last measured duration of the same ``--kind`` (``--state-dir``).
While it runs, the unit is stopped (SIGTERM to its process group, SIGKILL after a minute) when the window
closes plus ``grace_min`` or when the pause file appears; it is then put back at the end of the tsp queue with
the same command line and this job exits 75. ``touch ~/.offhours_pause`` therefore frees the GPU within a
minute and holds the queue until the file is removed. A unit restarts from scratch when requeued, so units
must be idempotent (baseline.sh skips finished training; an evaluation reruns all its seeds).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path

import yaml

STEP = timedelta(minutes=1)
HORIZON = timedelta(days=8)
EXIT_REQUEUED = 75


@dataclass
class Config:
    weekday_start: int = 21  # hour the weekday window opens
    weekday_end: int = 8  # hour it closes the next morning
    weekend_all_day: bool = True
    holidays: list = field(default_factory=list)  # "YYYY-MM-DD", treated like a weekend day
    margin_min: int = 10  # free time required after the estimated end
    grace_min: int = 30  # a running unit may overrun the window by this much
    gpu_busy_mib: int = 2048  # other compute processes holding more GPU memory than this block the start

    @classmethod
    def load(cls, path: Path) -> Config:
        if not Path(path).exists():
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown off-hours settings {sorted(unknown)} in {path}")
        return cls(**data)

    def allowed(self, when: datetime) -> bool:
        day_off = when.weekday() >= 5 or when.strftime("%Y-%m-%d") in {str(h) for h in self.holidays}
        if day_off and self.weekend_all_day:
            return True
        h = when.hour
        if self.weekday_start > self.weekday_end:
            return h >= self.weekday_start or h < self.weekday_end
        return self.weekday_start <= h < self.weekday_end


def window_end(now: datetime, cfg: Config) -> datetime | None:
    """When the window ``now`` is in closes (to the minute), None outside a window."""
    if not cfg.allowed(now):
        return None
    t = now.replace(second=0, microsecond=0)
    while cfg.allowed(t) and t - now < HORIZON:
        t += STEP
    return t


def next_window_start(now: datetime, cfg: Config) -> datetime:
    if cfg.allowed(now):
        return now
    t = now.replace(second=0, microsecond=0)
    while not cfg.allowed(t):
        t += STEP
        if t - now > HORIZON:
            raise ValueError("the off-hours window never opens; check the config")
    return t


def can_start(now: datetime, end: datetime | None, estimate_min: float, cfg: Config) -> bool:
    if end is None:
        return False
    return now + timedelta(minutes=estimate_min + cfg.margin_min) <= end


class Durations:
    """Measured minutes per kind of unit, to estimate the next one of that kind."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def estimate(self, kind: str, given_min: float) -> float:
        last = self._load().get(kind)
        return max(given_min, 1.1 * last) if last is not None else given_min

    def record(self, kind: str, minutes: float) -> None:
        data = self._load()
        data[kind] = minutes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1))


def requeue_command(label: str, argv: list[str], python: str = sys.executable) -> list[str]:
    return ["tsp", "-L", label, python, *argv]


def busy_from_smi(text: str, threshold_mib: float) -> bool:
    """``nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`` output: whether the
    compute processes hold more than ``threshold_mib`` together. Small resident viewers stay below it."""
    total = 0.0
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            try:
                total += float(parts[1])
            except ValueError:
                total += threshold_mib + 1  # unknown usage counts as busy
    return total > threshold_mib


def gpu_busy(threshold_mib: float) -> bool:
    """Someone else computes on the GPU (this job has not started its unit yet)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return busy_from_smi(out, threshold_mib)


def _stop(proc: subprocess.Popen, kill_after_s: float) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=kill_after_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_until(cmd: list[str], deadline: datetime, pause_file: Path, poll_s: float = 30.0,
              kill_after_s: float = 60.0) -> tuple[str, int | None]:
    """Run ``cmd`` in its own process group; ("done", exit code), or ("cut", None) when the deadline passes
    or the pause file appears first."""
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        while True:
            try:
                return "done", proc.wait(timeout=poll_s)
            except subprocess.TimeoutExpired:
                pass
            if datetime.now() >= deadline or Path(pause_file).exists():
                _stop(proc, kill_after_s)
                # Leftover members of the group (a child that outlived its parent) go too.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return "cut", None
    except BaseException:
        _stop(proc, kill_after_s)
        raise


def notify(script: str | None, message: str) -> None:
    print(message, flush=True)
    if script and Path(os.path.expanduser(script)).exists():
        subprocess.run([os.path.expanduser(script), message], check=False)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: offhours.py [options] -- <command ...>")
    split = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, help="kind of unit, for duration estimates (e.g. act_train)")
    p.add_argument("--estimate-min", type=float, required=True)
    p.add_argument("--label", default=None, help="tsp label when requeued (default: --kind)")
    p.add_argument("--config", default="~/.config/univtac-offhours.yaml")
    p.add_argument("--pause-file", default="~/.offhours_pause")
    p.add_argument("--state-dir", default="~/.local/state/univtac-offhours")
    p.add_argument("--notify", default="~/.claude/scripts/discord-notify.sh")
    p.add_argument("--poll-s", type=float, default=60.0)
    args = p.parse_args(argv[:split])
    cmd = argv[split + 1:]
    if not cmd:
        raise SystemExit("no command after --")

    cfg = Config.load(Path(os.path.expanduser(args.config)))
    pause = Path(os.path.expanduser(args.pause_file))
    durations = Durations(Path(os.path.expanduser(args.state_dir)) / "durations.json")
    label = args.label or args.kind
    estimate = durations.estimate(args.kind, args.estimate_min)

    while True:
        now = datetime.now()
        end = window_end(now, cfg)
        if not pause.exists() and can_start(now, end, estimate, cfg) and not gpu_busy(cfg.gpu_busy_mib):
            break
        time.sleep(args.poll_s)

    notify(args.notify, f"[offhours] start {label} (est. {estimate:.0f} min, window until {end:%a %H:%M})")
    t0 = time.monotonic()
    status, code = run_until(cmd, end + timedelta(minutes=cfg.grace_min), pause, poll_s=min(args.poll_s, 30.0))
    minutes = (time.monotonic() - t0) / 60
    if status == "cut":
        subprocess.run(requeue_command(label, [str(Path(__file__).resolve()), *argv]), check=False)
        notify(args.notify, f"[offhours] cut {label} after {minutes:.0f} min, requeued")
        return EXIT_REQUEUED
    if code == 0:
        durations.record(args.kind, minutes)
    notify(args.notify, f"[offhours] done {label}: exit {code} after {minutes:.0f} min")
    return code


if __name__ == "__main__":
    sys.exit(main())
