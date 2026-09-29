import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from envs.utils.replay_record import (
    episode_output_paths,
    hdf5_dataset_paths,
    merge_missing_observations,
    missing_observations,
    redirect_episode_output,
    source_metadata_entry,
    write_metadata_entry,
)


def write_hdf5(path: Path, datasets: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in datasets.items():
            f.create_dataset(key, data=value)
    return path


def source_episode(tmp_path, frames=4):
    return write_hdf5(
        tmp_path / "source" / "hdf5" / "3.hdf5",
        {
            "embodiment/joint": np.arange(frames * 9, dtype=np.float32).reshape(frames, 9),
            "tactile/left_tactile/pose": np.ones((frames, 7), dtype=np.float32),
        },
    )


# --- where the replay writes ----------------------------------------------------------


def test_episode_output_paths_keep_the_replayed_part_apart_from_the_dataset(tmp_path):
    cache, replayed, merged = episode_output_paths(tmp_path, 7)
    assert cache == tmp_path / ".cache" / "7"
    assert replayed == tmp_path / ".replayed" / "7.hdf5"
    assert merged == tmp_path / "hdf5" / "7.hdf5"


def test_redirect_episode_output_points_the_task_at_the_replayed_file(tmp_path):
    task = SimpleNamespace(tmp_save_dir=Path("old"), save_path=Path("old.hdf5"), save_count=12)
    redirect_episode_output(task, tmp_path, 3)
    assert task.tmp_save_dir == tmp_path / ".cache" / "3"
    assert task.save_path == tmp_path / ".replayed" / "3.hdf5"
    assert task.save_count == 0


# --- what the replay contributes ------------------------------------------------------


def test_hdf5_dataset_paths_lists_every_leaf(tmp_path):
    assert hdf5_dataset_paths(source_episode(tmp_path)) == {
        "embodiment/joint",
        "tactile/left_tactile/pose",
    }


def test_missing_observations_keeps_only_leaves_the_source_lacks():
    observation = {
        "embodiment": {"joint": np.zeros(9), "ee": np.zeros(7)},
        "tactile": {"left_tactile": {"pose": np.zeros(7), "force": np.ones(3), "torque": np.ones(3)}},
        "step": 5,
    }
    source = {"embodiment/joint", "embodiment/ee", "tactile/left_tactile/pose", "step"}
    kept = missing_observations(observation, source)
    assert set(kept) == {"tactile"}
    assert set(kept["tactile"]["left_tactile"]) == {"force", "torque"}


def test_missing_observations_is_empty_when_the_source_has_everything():
    assert missing_observations({"embodiment": {"joint": np.zeros(9)}}, {"embodiment/joint"}) == {}


# --- the merged episode ---------------------------------------------------------------


def test_merge_leaves_the_source_data_untouched_and_adds_the_replayed_part(tmp_path):
    source = source_episode(tmp_path)
    force = np.arange(12, dtype=np.float32).reshape(4, 3)
    replayed = write_hdf5(tmp_path / "out" / ".replayed" / "3.hdf5", {"tactile/left_tactile/force": force})
    merged = tmp_path / "out" / "hdf5" / "3.hdf5"

    added = merge_missing_observations(source, replayed, merged)

    assert added == ["tactile/left_tactile/force"]
    with h5py.File(merged, "r") as m, h5py.File(source, "r") as s:
        np.testing.assert_array_equal(m["embodiment/joint"][()], s["embodiment/joint"][()])
        np.testing.assert_array_equal(m["tactile/left_tactile/pose"][()], s["tactile/left_tactile/pose"][()])
        np.testing.assert_array_equal(m["tactile/left_tactile/force"][()], force)
    with h5py.File(source, "r") as s:
        assert "tactile/left_tactile/force" not in s


def test_merge_never_replaces_a_dataset_the_source_already_has(tmp_path):
    source = source_episode(tmp_path)
    replayed = write_hdf5(
        tmp_path / "out" / ".replayed" / "3.hdf5",
        {
            "embodiment/joint": np.full((4, 9), -1.0, dtype=np.float32),
            "tactile/left_tactile/force": np.zeros((4, 3), dtype=np.float32),
        },
    )
    merged = tmp_path / "out" / "hdf5" / "3.hdf5"
    added = merge_missing_observations(source, replayed, merged)
    assert added == ["tactile/left_tactile/force"]
    with h5py.File(merged, "r") as m, h5py.File(source, "r") as s:
        np.testing.assert_array_equal(m["embodiment/joint"][()], s["embodiment/joint"][()])


def test_merge_rejects_a_replay_with_a_different_frame_count(tmp_path):
    source = source_episode(tmp_path, frames=4)
    replayed = write_hdf5(
        tmp_path / "out" / ".replayed" / "3.hdf5",
        {"tactile/left_tactile/force": np.zeros((5, 3), dtype=np.float32)},
    )
    merged = tmp_path / "out" / "hdf5" / "3.hdf5"
    with pytest.raises(ValueError, match="frames"):
        merge_missing_observations(source, replayed, merged)
    assert not merged.exists()


def test_merge_rejects_a_replay_that_adds_nothing(tmp_path):
    source = source_episode(tmp_path)
    replayed = write_hdf5(
        tmp_path / "out" / ".replayed" / "3.hdf5",
        {"embodiment/joint": np.zeros((4, 9), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="nothing"):
        merge_missing_observations(source, replayed, tmp_path / "out" / "hdf5" / "3.hdf5")


# --- metadata -------------------------------------------------------------------------


def test_source_metadata_entry_reads_the_dataset_root_next_to_the_hdf5_folder(tmp_path):
    root = tmp_path / "clean51"
    (root / "hdf5").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"4": {"rotate": 3.14, "seed": 4}}))
    assert source_metadata_entry(root / "hdf5" / "4.hdf5", 4) == {"rotate": 3.14, "seed": 4}


def test_source_metadata_entry_reads_a_flat_dataset_dir(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"1": {"rotate": 0.0}}))
    assert source_metadata_entry(tmp_path / "1.hdf5", 1) == {"rotate": 0.0}


def test_source_metadata_entry_is_empty_without_metadata(tmp_path):
    assert source_metadata_entry(tmp_path / "1.hdf5", 1) == {}


def test_write_metadata_entry_merges_seeds_into_one_file(tmp_path):
    write_metadata_entry(tmp_path, 0, {"rotate": 0.0, "seed": 0})
    write_metadata_entry(tmp_path, 1, {"rotate": 3.14, "seed": 1})
    write_metadata_entry(tmp_path, 0, {"rotate": 0.0, "seed": 0, "replay": {"max_ee_error": 0.001}})
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert set(saved) == {"0", "1"}
    assert saved["0"]["replay"]["max_ee_error"] == 0.001
    assert saved["1"]["rotate"] == 3.14
