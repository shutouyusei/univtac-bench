"""RDP's control loop (official real_runner.py index logic, latency_step 0 in the synchronous simulator): a new
latent every 16 steps from the frames at t-2 and t, and every step the tokenizer's GRU decoder re-run over the
tactile from 3 steps before the chunk start up to now, its last output executed, arm joints made absolute."""

import numpy as np
import pytest
import torch

from conftest import write_demo
from rdp_univtac import build, controller, prepare, train
from test_train import TOY


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rdp")
    demos = tmp / "hdf5"
    demos.mkdir()
    for i, T in enumerate((30, 26)):
        write_demo(demos / f"{i}.hdf5", T, i)
    cache = tmp / "cache"
    prepare.build_cache(prepare.demo_paths(demos, 2), cameras=["head"], out=cache)
    pca = tmp / "pca.npz"
    prepare.fit_pooled_pca([prepare.demo_paths(demos, 2)]).save(pca)
    cfg = build.load_config(override=TOY)
    train.train_at(cfg, cache, pca, tmp / "out", "cpu")
    return train.train_ldp(cfg, cache, pca, tmp / "out", "cpu")


def step_inputs(k):
    frames = {"cam_head": np.full((240, 320, 3), k, np.uint8)}
    joint = np.arange(8, dtype=np.float32) + 10 * k
    emb = {"tactile_left": np.full(15, k, np.float32), "tactile_right": np.full(15, -k, np.float32)}
    return frames, joint, emb


def test_obs_frames_pick_t_minus_2_and_t_with_start_padding():
    assert controller.obs_indices(0) == [0, 0]
    assert controller.obs_indices(1) == [0, 1]
    assert controller.obs_indices(7) == [5, 7]


def test_tactile_window_from_3_before_the_chunk_start_to_now():
    assert controller.tactile_indices(chunk_start=16, now=16) == [13, 14, 15, 16]
    assert controller.tactile_indices(chunk_start=16, now=20) == [13, 14, 15, 16, 17, 18, 19, 20]
    assert controller.tactile_indices(chunk_start=0, now=2) == [0, 0, 0, 0, 1, 2]


def test_one_latent_per_16_steps_and_growing_decoder_windows(bundle, monkeypatch):
    c = controller.RDPController.load(bundle, device="cpu")
    calls = {"latent": 0, "ext": []}
    real_predict, real_decode = c.policy.predict_action, c.policy.predict_from_latent_action

    def count_predict(*a, **k):
        calls["latent"] += 1
        return real_predict(*a, **k)

    def record_decode(latent, ext_obs, last_step, ratio, **k):
        calls["ext"].append((last_step, ext_obs["tactile_left"][0, -1, 0].item(), ext_obs["tactile_left"].shape[1]))
        return real_decode(latent, ext_obs, last_step, ratio, **k)

    monkeypatch.setattr(c.policy, "predict_action", count_predict)
    monkeypatch.setattr(c.policy, "predict_from_latent_action", record_decode)
    c.reset()
    for k in range(34):
        a = c.act(*step_inputs(k))
        assert a.shape == (8,) and np.all(np.isfinite(a))
    assert calls["latent"] == 3  # steps 0, 16, 32
    assert [e[0] for e in calls["ext"][:3]] == [4, 5, 6]
    assert calls["ext"][16][0] == 4 and calls["ext"][31][0] == 19
    assert all(newest == k for k, (_, newest, _) in enumerate(calls["ext"]))  # newest tactile is the current one
    assert all(n == ext for ext, _, n in calls["ext"])


def test_actions_are_absolute_arm_joints_and_absolute_gripper(bundle, monkeypatch):
    c = controller.RDPController.load(bundle, device="cpu")
    rel = torch.zeros(1, 40, 8)
    rel[..., :7] = 0.5
    rel[..., 7] = 0.02

    def fake_decode(latent, ext_obs, last_step, ratio, **k):
        return {"action_pred": rel[:, :last_step], "action": rel[:, 3:last_step]}

    monkeypatch.setattr(c.policy, "predict_from_latent_action", fake_decode)
    c.reset()
    frames, joint, emb = step_inputs(0)
    a = c.act(frames, joint, emb)
    assert np.allclose(a[:7], joint[:7] + 0.5) and a[7] == pytest.approx(0.02)
    # inside the chunk the base stays the chunk start's arm joints
    a = c.act(*step_inputs(1))
    assert np.allclose(a[:7], joint[:7] + 0.5)


def test_reset_clears_the_history(bundle):
    c = controller.RDPController.load(bundle, device="cpu")
    c.act(*step_inputs(0))
    c.reset()
    assert c.step == 0 and not c.history["tactile_left"]
