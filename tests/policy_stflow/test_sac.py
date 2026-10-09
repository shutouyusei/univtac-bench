"""SAC on the fast side of a slow-fast SFP checkpoint: the controller, the replay buffer and the update."""

import dataclasses

import numpy as np
import pytest
import torch

stflow = pytest.importorskip("stflow")

from stflow.config import ModelConfig  # noqa: E402
from stflow.methods.sfp import SFPController  # noqa: E402
from stflow.model.policy import FlowPolicy  # noqa: E402

from policy.stflow.sac import (  # noqa: E402
    SAC, ReplayBuffer, SACConfig, SACController, gaussian_log_prob, pooled_reading,
)

SIZE = 32
CHUNK = 6
REFRESH = 2
A = 8


@pytest.fixture
def model():
    if "tactile_prediction" not in {f.name for f in dataclasses.fields(ModelConfig)}:
        pytest.skip("this stflow has no tactile prediction")
    cfg = ModelConfig(
        image_size=SIZE, pretrained_backbone=False, d_model=32, n_heads=4, n_layers=1, dim_feedforward=64,
        dropout=0.0, chunk_size=CHUNK, tactile_adapter=True, tactile_prediction=True, slow_tactile=True,
    )
    torch.manual_seed(0)
    policy = FlowPolicy(cfg).eval()
    for adapter in policy.expert.adapters:  # zero-initialised gates would hide the fast side
        for p in adapter.parameters():
            torch.nn.init.normal_(p, std=0.5)
    return policy


def observations(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{
        "images": {"head": torch.rand(1, 3, SIZE, SIZE, generator=g)},
        "tactile": {t: torch.rand(1, 3, SIZE, SIZE, generator=g) for t in ("left", "right")},
        "state": torch.rand(1, 8, generator=g),
    } for _ in range(n)]


def sac_config(**kw):
    return SACConfig(scale=[0.5] * A, batch_size=8, warmup_episodes=0, hidden=32, **kw)


def controller(model, explore, cfg=None, seed=0):
    cfg = cfg or sac_config()
    learner = SAC(model, cfg)
    return SACController(model, learner, CHUNK, REFRESH, explore=explore, seed=seed), learner


def test_without_exploration_the_controller_acts_like_the_sfp_controller(model):
    sfp = SFPController(model, seed=0, n_action_steps=CHUNK, tactile_refresh=REFRESH)
    sac, _ = controller(model, explore=False)
    with torch.no_grad():
        for obs in observations(2 * CHUNK + 1):
            expected = sfp.act(obs)
            action, _ = sac.act(obs)
            assert torch.allclose(action, expected, atol=1e-5)


def test_records_carry_the_fast_residual_and_rewards_only_at_block_starts(model):
    sac, _ = controller(model, explore=True)
    with torch.no_grad():
        records = [sac.act(obs)[1] for obs in observations(2 * CHUNK)]
    queries = [r["q"] for r in records]
    assert queries == list(range(CHUNK)) * 2
    # A reward arrives at a block start inside a chunk (the reading the slow side forecast), never at a
    # chunk start: there the slow side sees the reading itself.
    assert [r["reward"] is not None for r in records] == [q % REFRESH == 0 and q > 0 for q in queries]
    assert all(r["reward"] <= 0 for r in records if r["reward"] is not None)
    assert [r["new_chunk"] for r in records] == [q == 0 for q in queries]
    assert [r["new_block"] for r in records] == [q % REFRESH == 0 for q in queries]
    for r in records:
        assert r["action"].shape == r["mean"].shape == r["v_slow"].shape == r["x"].shape == (A,)
        assert r["hidden"].shape == (32,) and r["reading"] is None or r["reading"].shape == (2, 32)


def test_the_reward_forecast_follows_the_slow_plan_not_the_fast_or_the_noise(model):
    obs = observations(CHUNK)

    def rewards(adapter_std, seed):
        torch.manual_seed(1)
        for adapter in model.expert.adapters:
            for p in adapter.parameters():
                torch.nn.init.normal_(p, std=adapter_std)
        sac, _ = controller(model, explore=True, seed=seed)
        with torch.no_grad():
            return [sac.act(o)[1]["reward"] for o in obs]

    a, b = rewards(0.5, 0), rewards(0.1, 7)
    assert [x is None for x in a] == [x is None for x in b]
    assert np.allclose([x for x in a if x is not None], [x for x in b if x is not None], atol=1e-6)


def test_gaussian_log_prob_matches_torch():
    mean, log_std, a = torch.randn(5, A), torch.randn(A) * 0.3, torch.randn(5, A)
    expected = torch.distributions.Normal(mean, log_std.exp()).log_prob(a).sum(-1)
    assert torch.allclose(gaussian_log_prob(a, mean, log_std), expected, atol=1e-5)


def test_pooled_reading_is_the_layer_normed_mean_over_each_fingertips_tokens(model):
    tokens = model.tactile_tokens(observations(1)[0]["tactile"])
    pooled = pooled_reading(model, tokens)
    assert pooled.shape == (1, 2, 32)
    assert torch.allclose(pooled.mean(-1), torch.zeros(1, 2), atol=1e-5)


def fill(buffer, sac, episodes, steps):
    obs = observations(steps)
    for _ in range(episodes):
        sac.reset()
        buffer.start_episode()
        with torch.no_grad():
            for o in obs:
                buffer.add(sac.act(o)[1])
        buffer.end_episode()


def test_buffer_puts_each_reward_on_the_step_before_and_never_samples_a_last_step(model, tmp_path):
    sac, _ = controller(model, explore=True)
    buffer = ReplayBuffer()
    fill(buffer, sac, episodes=2, steps=2 * CHUNK + 1)
    ep = buffer.episodes[0]
    n = 2 * CHUNK + 1
    assert len(ep["q"]) == n and ep["has_next"].tolist() == [True] * (n - 1) + [False]
    rewarded = [k for k in range(n) if ep["r"][k] != 0]
    assert rewarded == [k - 1 for k in range(n) if k % CHUNK % REFRESH == 0 and k % CHUNK > 0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        e, k = buffer.sample(8, rng)
        assert all(buffer.episodes[i]["has_next"][j] for i, j in zip(e, k))
    buffer.save_episode(0, tmp_path / "ep0.npz")
    loaded = ReplayBuffer()
    loaded.load_dir(tmp_path)
    for key, value in buffer.episodes[0].items():
        assert np.array_equal(loaded.episodes[0][key], value), key


def test_update_moves_the_adapters_and_the_critic_only(model):
    cfg = sac_config()
    sac, learner = controller(model, explore=True, cfg=cfg)
    buffer = ReplayBuffer()
    fill(buffer, sac, episodes=2, steps=2 * CHUNK + 1)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("expert.adapters.")}
    adapters = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("expert.adapters.")}
    critic = [p.detach().clone() for p in learner.critic.parameters()]
    stats = learner.update(buffer, n_updates=3, rng=np.random.default_rng(0))
    assert {"critic_loss", "actor_loss", "alpha", "q_mean", "entropy"} <= set(stats)
    assert all(np.isfinite(v) for v in stats.values())
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    assert any(not torch.equal(p, adapters[n]) for n, p in model.named_parameters() if n in adapters)
    assert any(not torch.equal(p, c) for p, c in zip(learner.critic.parameters(), critic))


def test_learner_state_round_trips(model, tmp_path):
    cfg = sac_config()
    sac, learner = controller(model, explore=True, cfg=cfg)
    buffer = ReplayBuffer()
    fill(buffer, sac, episodes=1, steps=CHUNK + 1)
    learner.update(buffer, n_updates=2, rng=np.random.default_rng(0))
    learner.save(tmp_path / "learner.pt")
    other_model = FlowPolicy(model.cfg).eval()
    other = SAC(other_model, cfg)
    other.load(tmp_path / "learner.pt")
    for (n, p), q in zip(model.expert.adapters.named_parameters(), other_model.expert.adapters.parameters()):
        assert torch.equal(p, q), n
    assert torch.equal(learner.log_std, other.log_std) and learner.updates == other.updates
