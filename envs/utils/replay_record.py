"""Where a replayed episode's re-recorded observations and metadata go.

``scripts/replay.py --record-dir`` re-records every replayed frame with the
observation types of the active task config (for example the contact ``force``
and ``torque`` that the original collection did not save). The output mirrors
the collect layout, ``<record_dir>/hdf5/<seed>.hdf5`` plus one
``<record_dir>/metadata.json`` keyed by seed, so the converters that read a
collected dataset read a re-recorded one unchanged.
"""

import json
from pathlib import Path


def episode_output_paths(record_dir: Path, seed: int) -> tuple[Path, Path]:
    """``(.cache/<seed>, hdf5/<seed>.hdf5)`` under ``record_dir``, as BaseTask lays them out."""
    record_dir = Path(record_dir)
    return record_dir / ".cache" / str(seed), record_dir / "hdf5" / f"{seed}.hdf5"


def redirect_episode_output(task, record_dir: Path, seed: int) -> None:
    """Make ``task.save_observations`` / ``task.save_to_hdf5`` write this seed under ``record_dir``."""
    task.tmp_save_dir, task.save_path = episode_output_paths(record_dir, seed)
    task.save_count = 0


def source_metadata_entry(data_path: Path, seed: int) -> dict:
    """The collected dataset's metadata for ``seed`` (``metadata.json`` beside the ``hdf5`` folder or the file)."""
    data_path = Path(data_path)
    for root in (data_path.parent.parent, data_path.parent):
        metadata_file = root / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                return dict(json.load(f).get(str(seed), {}))
    return {}


def write_metadata_entry(record_dir: Path, seed: int, entry: dict) -> Path:
    """Merge ``entry`` under key ``str(seed)`` into ``<record_dir>/metadata.json``."""
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = record_dir / "metadata.json"
    all_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    all_metadata[str(seed)] = entry
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=4)
    return metadata_file
