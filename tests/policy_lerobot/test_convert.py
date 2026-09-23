"""HDF5 -> LeRobot converter: the pure transforms (pairing, decoding, schema, instruction lookup, episode encoding)."""

import json

import cv2
import numpy as np
import pytest

from policy.lerobot.convert import hdf5, pipeline, schema, transforms


def test_episode_files_sorted_numerically(tmp_path):
    for stem in ("10", "2", "0"):
        (tmp_path / f"{stem}.hdf5").touch()
    assert [p.stem for p in hdf5.episode_files(tmp_path)] == ["0", "2", "10"]
    assert len(hdf5.select_episode_files(tmp_path, 2)) == 2
    with pytest.raises(ValueError):
        hdf5.select_episode_files(tmp_path, 4)


def test_instruction_for_uses_first_seen_entry(tmp_path):
    (tmp_path / "insert_hole.json").write_text(json.dumps({"seen": ["Insert it.", "Other"], "unseen": ["x"]}))
    assert schema.instruction_for("insert_hole", tmp_path) == "Insert it."
    assert schema.instruction_for("lift_can", tmp_path) == "lift can"


def test_visual_cameras_follow_task_settings(tmp_path):
    settings = tmp_path / "task_settings.json"
    settings.write_text(json.dumps({"lift_can": {"camera_type": "all"}, "insert_hole": {"camera_type": "head"}}))
    assert schema.visual_cameras("lift_can", settings) == ("head", "wrist")
    assert schema.visual_cameras("insert_hole", settings) == ("head",)
    assert schema.visual_cameras("unknown", settings) == ("head",)


def test_split_transitions_pairs_next_joint_as_action():
    joint = np.arange(4 * 9, dtype=np.float32).reshape(4, 9)
    state, action = transforms.split_transitions(joint)
    assert state.shape == (3, 8) and action.shape == (3, 8)
    np.testing.assert_array_equal(state, joint[:-1, :8])
    np.testing.assert_array_equal(action, joint[1:, :8])
    with pytest.raises(ValueError):
        transforms.split_transitions(joint[:1])


def test_decode_frames_roundtrips_png_in_bgr():
    img = np.zeros((8, 6, 3), np.uint8)
    img[..., 0] = 200  # blue in BGR
    ok, buf = cv2.imencode(".png", img)
    assert ok
    frames = hdf5.decode_frames(np.array([buf.tobytes(), buf.tobytes()], dtype=object))
    assert frames.shape == (2, 8, 6, 3)
    np.testing.assert_array_equal(frames[0], img)
    assert transforms.bgr_to_rgb(frames)[0, 0, 0, 2] == 200


def test_resize_frames_to_square():
    frames = np.random.randint(0, 255, (2, 270, 480, 3), np.uint8)
    out = transforms.resize_frames(frames, 256)
    assert out.shape == (2, 256, 256, 3) and out.dtype == np.uint8
    assert transforms.resize_frames(out, 256) is out


def test_stack_fingertips_concatenates_per_frame():
    left = np.ones((3, 4), np.float32)
    right = np.zeros((3, 4), np.float32)
    out = transforms.stack_fingertips([left, right])
    assert out.shape == (3, 8)
    assert out[0, :4].sum() == 4 and out[0, 4:].sum() == 0


def test_build_features_schema():
    feats = schema.build_features({"observation.images.head": (256, 256)}, tactile_channels=1536, use_videos=False)
    assert feats["observation.images.head"] == {
        "dtype": "image",
        "shape": (3, 256, 256),
        "names": ["channel", "height", "width"],
    }
    assert feats["observation.state"]["shape"] == (8,)
    assert feats["action"]["shape"] == (8,)
    assert feats["observation.environment_state"]["shape"] == (1536,)
    assert schema.build_features({}, 4, use_videos=True) == schema.build_features({}, 4, use_videos=False)


def _episode(T=4):
    return {
        "joint": np.arange(T * 9, dtype=np.float32).reshape(T, 9),
        "cameras": {"head": np.random.randint(0, 255, (T, 270, 480, 3), np.uint8)},
        "tactile": {
            "left": np.random.randint(0, 255, (T, 240, 320, 3), np.uint8),
            "right": np.random.randint(0, 255, (T, 240, 320, 3), np.uint8),
        },
    }


def test_encode_episode_builds_dataset_rows_with_and_without_tactile_images():
    ep = _episode()
    calls = []

    def fake_embed(frames_bgr):
        calls.append(frames_bgr.shape)
        return np.full((len(frames_bgr), 3), 0.5, np.float32)

    frames = pipeline.encode_episode(ep, ("head",), fake_embed, image_size=64, tactile_images=False)
    assert len(frames) == 3
    assert frames.state.shape == (3, 8) and frames.action.shape == (3, 8)
    assert frames.env_state.shape == (3, 6)
    assert list(frames.images) == ["observation.images.head"]
    assert frames.images["observation.images.head"].shape == (3, 64, 64, 3)
    assert calls == [(3, 240, 320, 3), (3, 240, 320, 3)]  # left then right, last frame dropped

    with_tac = pipeline.encode_episode(ep, ("head",), fake_embed, image_size=64, tactile_images=True)
    assert set(with_tac.images) == {
        "observation.images.head",
        "observation.images.tactile_left",
        "observation.images.tactile_right",
    }
    assert with_tac.images["observation.images.tactile_left"].shape == (3, 240, 320, 3)
    assert pipeline.image_shapes_for(ep, ("head",), 64, True)["observation.images.tactile_left"] == (240, 320)


def test_write_episode_adds_every_row_then_saves():
    class FakeDataset:
        def __init__(self):
            self.frames, self.saved = [], 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1

    frames = pipeline.encode_episode(_episode(), ("head",), lambda x: np.zeros((len(x), 2), np.float32), 32, False)
    ds = FakeDataset()
    pipeline.write_episode(ds, frames, "Insert.")
    assert ds.saved == 1 and len(ds.frames) == 3
    assert set(ds.frames[0]) == {"observation.state", "action", "observation.environment_state", "task", "observation.images.head"}
    assert ds.frames[0]["task"] == "Insert."


def test_prepare_output_dir_respects_overwrite(tmp_path):
    out = tmp_path / "ds"
    out.mkdir()
    (out / "x").touch()
    with pytest.raises(FileExistsError):
        pipeline.prepare_output_dir(out, overwrite=False)
    assert not (pipeline.prepare_output_dir(out, overwrite=True)).exists()
