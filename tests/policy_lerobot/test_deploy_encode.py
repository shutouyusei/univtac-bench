"""Bridge encoding: wire round trip, observation trimming on the Isaac side, batch building on the lerobot side."""

import numpy as np
import pytest
import torch

from policy.lerobot import deploy_policy as dp
from policy.lerobot import wire
from policy.lerobot.server import build_batch


def test_wire_roundtrip_preserves_arrays_and_scalars():
    payload = {
        "img": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "joint": torch.arange(8, dtype=torch.float32),
        "text": "insert",
        "n": 3,
        "nested": [np.float32(1.5), None, {"flag": True}],
    }
    out = wire.from_wire(wire.to_wire(payload))
    np.testing.assert_array_equal(out["img"], payload["img"])
    assert out["img"].dtype == np.uint8 and out["img"].flags["WRITEABLE"]
    np.testing.assert_array_equal(out["joint"], np.arange(8, dtype=np.float32))
    assert out["text"] == "insert" and out["n"] == 3
    assert out["nested"] == [1.5, None, {"flag": True}]


def _sim_observation(dtype=torch.uint8, tactile_key="left_tactile"):
    def img(h, w):
        x = torch.randint(0, 255, (h, w, 3), dtype=torch.uint8)
        return x if dtype == torch.uint8 else x.float() / 255.0

    return {
        "observation": {"head": {"rgb": img(270, 480)}, "wrist": {"rgb": img(270, 480)}},
        "tactile": {
            tactile_key: {"rgb_marker": img(240, 320), "depth": torch.zeros(240, 320)},
            tactile_key.replace("left", "right"): {"rgb_marker": img(240, 320)},
        },
        "embodiment": {"joint": torch.arange(9, dtype=torch.float32), "ee": torch.zeros(7)},
    }


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_select_observation_keeps_only_bridge_inputs(dtype):
    obs = _sim_observation(dtype)
    out = dp.select_observation(obs, ("head",))
    assert set(out) == {"images", "tactile", "joint"}
    assert list(out["images"]) == ["head"]
    assert out["images"]["head"].shape == (270, 480, 3) and out["images"]["head"].dtype == np.uint8
    assert set(out["tactile"]) == {"left", "right"}
    assert out["tactile"]["left"].shape == (240, 320, 3) and out["tactile"]["left"].dtype == np.uint8
    np.testing.assert_array_equal(out["joint"], np.arange(8, dtype=np.float32))
    expected = obs["observation"]["head"]["rgb"]
    expected = expected.numpy() if dtype == torch.uint8 else np.rint(expected.numpy() * 255).astype(np.uint8)
    np.testing.assert_array_equal(out["images"]["head"], expected)


def test_select_observation_accepts_gsmini_alias_and_all_cameras():
    out = dp.select_observation(_sim_observation(tactile_key="left_gsmini"), ("head", "wrist"))
    assert list(out["images"]) == ["head", "wrist"]
    assert set(out["tactile"]) == {"left", "right"}


def test_select_observation_reports_missing_streams():
    obs = _sim_observation()
    del obs["tactile"]["right_tactile"]
    with pytest.raises(KeyError):
        dp.select_observation(obs, ("head",))


def test_cameras_for_task(tmp_path):
    settings = tmp_path / "task_settings.json"
    settings.write_text('{"lift_can": {"camera_type": "all"}}')
    assert dp.cameras_for_task("lift_can", settings) == ("head", "wrist")
    assert dp.cameras_for_task("insert_hole", settings) == ("head",)


def test_build_batch_matches_dataset_contract():
    obs = dp.select_observation(_sim_observation(), ("head",))
    seen = {}

    def fake_embed(images_bgr):
        seen["images"] = images_bgr
        return np.full((len(images_bgr), 4), 0.5, np.float32)

    batch = build_batch(obs, fake_embed, "Insert the tube.", image_size=256)
    assert set(batch) == {"observation.images.head", "observation.state", "observation.environment_state", "task"}
    img = batch["observation.images.head"]
    assert img.shape == (3, 256, 256) and img.dtype == torch.float32 and 0.0 <= img.min() and img.max() <= 1.0
    assert batch["observation.state"].shape == (8,)
    assert batch["observation.environment_state"].shape == (8,)
    assert batch["task"] == "Insert the tube."
    # encoder sees left then right, in BGR
    assert seen["images"].shape == (2, 240, 320, 3)
    np.testing.assert_array_equal(seen["images"][0][..., 0], obs["tactile"]["left"][..., 2])
