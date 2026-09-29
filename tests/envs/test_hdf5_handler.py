import h5py
import numpy as np

from envs.utils.data import HDF5Handler


def test_dict_to_hdf5_stores_a_column_of_empty_strings(tmp_path):
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as f:
        HDF5Handler().dict_to_hdf5(f, {"atom": {"tag": ["", "", ""]}})
    with h5py.File(path, "r") as f:
        tags = f["atom/tag"][()]
    assert tags.shape == (3,)
    assert all(t == b"" for t in tags)


def test_dict_to_hdf5_keeps_the_longest_string_intact(tmp_path):
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as f:
        HDF5Handler().dict_to_hdf5(f, {"tag": ["", "grasp", "move"], "force": [np.zeros(3), np.ones(3)]})
    with h5py.File(path, "r") as f:
        assert [t.decode() for t in f["tag"][()]] == ["", "grasp", "move"]
        assert f["force"].shape == (2, 3)
