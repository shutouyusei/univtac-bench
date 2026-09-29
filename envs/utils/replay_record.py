"""Fill observations a collected dataset lacks by replaying its episodes.

``scripts/replay.py --record-dir`` replays each episode's joint trajectory under
the active task config and saves, per replayed frame, only the observations
the source episode does not have (for example the contact ``force`` and
``torque``). The output episode is the source file plus those datasets: joints,
end-effector pose, images and tactile frames stay the collected ones, so a
policy trained on the filled dataset sees the same inputs and actions as one
trained on the source.

Layout under ``record_dir`` (the collect layout, so converters read it unchanged):

* ``hdf5/<seed>.hdf5``      source episode + the replayed datasets
* ``metadata.json``         source entry + replay result and tracking errors, per seed
* ``.replayed/<seed>.hdf5`` the replayed datasets alone, removed after the merge
"""

import json
import shutil
from pathlib import Path

import h5py


def episode_output_paths(record_dir: Path, seed: int) -> tuple[Path, Path, Path]:
    """``(.cache/<seed>, .replayed/<seed>.hdf5, hdf5/<seed>.hdf5)`` under ``record_dir``."""
    record_dir = Path(record_dir)
    return (
        record_dir / ".cache" / str(seed),
        record_dir / ".replayed" / f"{seed}.hdf5",
        record_dir / "hdf5" / f"{seed}.hdf5",
    )


def redirect_episode_output(task, record_dir: Path, seed: int) -> None:
    """Make ``task.save_observations`` / ``task.save_to_hdf5`` write this seed's replayed part."""
    task.tmp_save_dir, task.save_path, _merged = episode_output_paths(record_dir, seed)
    task.save_count = 0


def hdf5_dataset_paths(hdf5_path: Path) -> set[str]:
    """Paths of every dataset in the file, e.g. ``tactile/left_tactile/pose``."""
    paths = set()

    def visit(name, node):
        if isinstance(node, h5py.Dataset):
            paths.add(name)

    with h5py.File(hdf5_path, "r") as f:
        f.visititems(visit)
    return paths


def missing_observations(observation: dict, source_paths: set[str], prefix: str = "") -> dict:
    """The leaves of a nested observation whose ``a/b/c`` path is not in ``source_paths``."""
    kept = {}
    for key, value in observation.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            sub = missing_observations(value, source_paths, prefix=f"{path}/")
            if sub:
                kept[key] = sub
        elif path not in source_paths:
            kept[key] = value
    return kept


def merge_missing_observations(source: Path, replayed: Path, merged: Path) -> list[str]:
    """Write ``merged`` = ``source`` + the datasets of ``replayed`` that ``source`` lacks.

    Nothing of the source is replaced. Every added dataset must have one row per
    source frame, otherwise its rows would not belong to the source's frames.
    Returns the added dataset paths, sorted.
    """
    source, replayed, merged = Path(source), Path(replayed), Path(merged)
    added = sorted(hdf5_dataset_paths(replayed) - hdf5_dataset_paths(source))
    if not added:
        raise ValueError(f"{replayed} holds nothing that {source} lacks")
    with h5py.File(source, "r") as s, h5py.File(replayed, "r") as r:
        frames = s["embodiment/joint"].shape[0]
        for path in added:
            if r[path].shape[0] != frames:
                raise ValueError(
                    f"{path} has {r[path].shape[0]} frames, the source episode has {frames}"
                )
    merged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, merged)
    with h5py.File(merged, "a") as m, h5py.File(replayed, "r") as r:
        for path in added:
            r.copy(r[path], m.require_group(str(Path(path).parent)), name=Path(path).name)
    return added


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
