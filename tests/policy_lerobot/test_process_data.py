"""HDF5 -> LeRobot converter: the pure transforms (pairing, decoding, schema, instruction lookup)."""

import json

import cv2
import numpy as np
import pytest

from policy.lerobot import process_data as pd


def test_episode_files_sorted_numerically(tmp_path):
    for stem in ("10", "2", "0"):
        (tmp_path / f"{stem}.hdf5").touch()
    assert [p.stem for p in pd.episode_files(tmp_path)] == ["0", "2", "10"]


def test_instruction_for_uses_first_seen_entry(tmp_path):
    (tmp_path / "insert_hole.json").write_text(json.dumps({"seen": ["Insert it.", "Other"], "unseen": ["x"]}))
    assert pd.instruction_for("insert_hole", tmp_path) == "Insert it."
    assert pd.instruction_for("lift_can", tmp_path) == "lift can"


def test_visual_cameras_follow_task_settings(tmp_path):
    settings = tmp_path / "task_settings.json"
    settings.write_text(json.dumps({"lift_can": {"camera_type": "all"}, "insert_hole": {"camera_type": "head"}}))
    assert pd.visual_cameras("lift_can", settings) == ("head", "wrist")
    assert pd.visual_cameras("insert_hole", settings) == ("head",)
    assert pd.visual_cameras("unknown", settings) == ("head",)


def test_split_transitions_pairs_next_joint_as_action():
    joint = np.arange(4 * 9, dtype=np.float32).reshape(4, 9)
    state, action = pd.split_transitions(joint)
    assert state.shape == (3, 8) and action.shape == (3, 8)
    np.testing.assert_array_equal(state, joint[:-1, :8])
    np.testing.assert_array_equal(action, joint[1:, :8])
    with pytest.raises(ValueError):
        pd.split_transitions(joint[:1])


def test_decode_frames_roundtrips_png_in_bgr():
    img = np.zeros((8, 6, 3), np.uint8)
    img[..., 0] = 200  # blue in BGR
    ok, buf = cv2.imencode(".png", img)
    assert ok
    frames = pd.decode_frames(np.array([buf.tobytes(), buf.tobytes()], dtype=object))
    assert frames.shape == (2, 8, 6, 3)
    np.testing.assert_array_equal(frames[0], img)
    assert pd.bgr_to_rgb(frames)[0, 0, 0, 2] == 200


def test_resize_frames_to_square():
    frames = np.random.randint(0, 255, (2, 270, 480, 3), np.uint8)
    out = pd.resize_frames(frames, 256)
    assert out.shape == (2, 256, 256, 3) and out.dtype == np.uint8
    assert pd.resize_frames(out, 256) is out


def test_stack_fingertips_concatenates_per_frame():
    left = np.ones((3, 4), np.float32)
    right = np.zeros((3, 4), np.float32)
    out = pd.stack_fingertips([left, right])
    assert out.shape == (3, 8)
    assert out[0, :4].sum() == 4 and out[0, 4:].sum() == 0


def test_build_features_schema():
    feats = pd.build_features({"observation.images.head": (256, 256)}, tactile_channels=1536, use_videos=False)
    assert feats["observation.images.head"] == {
        "dtype": "image",
        "shape": (3, 256, 256),
        "names": ["channel", "height", "width"],
    }
    assert feats["observation.state"]["shape"] == (8,)
    assert feats["action"]["shape"] == (8,)
    assert feats["observation.environment_state"]["shape"] == (1536,)
    assert pd.build_features({}, 4, use_videos=True) == {
        k: v for k, v in pd.build_features({}, 4, use_videos=False).items()
    }
