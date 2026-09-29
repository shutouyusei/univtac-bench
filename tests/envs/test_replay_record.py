import json
from pathlib import Path
from types import SimpleNamespace

from envs.utils.replay_record import (
    episode_output_paths,
    redirect_episode_output,
    source_metadata_entry,
    write_metadata_entry,
)


def test_episode_output_paths_follow_the_collect_layout(tmp_path):
    cache, hdf5 = episode_output_paths(tmp_path, 7)
    assert cache == tmp_path / ".cache" / "7"
    assert hdf5 == tmp_path / "hdf5" / "7.hdf5"


def test_redirect_episode_output_points_the_task_at_the_record_dir(tmp_path):
    task = SimpleNamespace(tmp_save_dir=Path("old"), save_path=Path("old.hdf5"), save_count=12)
    redirect_episode_output(task, tmp_path, 3)
    assert task.tmp_save_dir == tmp_path / ".cache" / "3"
    assert task.save_path == tmp_path / "hdf5" / "3.hdf5"
    assert task.save_count == 0


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
