from types import SimpleNamespace

import pytest

from policy.lerobot.bridge.local_policy import Replanner, apply_n_action_steps


def chunk_config(**kw):
    return SimpleNamespace(type="smolvla", chunk_size=50, n_action_steps=50, **kw)


def streaming_config():
    return chunk_config(steps_per_block=5, block_size=5)


def test_none_keeps_the_checkpoint_value():
    config = chunk_config()
    assert apply_n_action_steps(config, None) is None
    assert config.n_action_steps == 50


def test_chunking_policy_takes_it_in_its_config():
    config = chunk_config()
    assert apply_n_action_steps(config, 25) is None
    assert config.n_action_steps == 25


@pytest.mark.parametrize("n", [0, 51])
@pytest.mark.parametrize("make", [chunk_config, streaming_config])
def test_rejects_values_outside_the_chunk(n, make):
    with pytest.raises(ValueError):
        apply_n_action_steps(make(), n)


def test_streaming_policy_leaves_its_config_and_replans_from_the_bridge():
    config = streaming_config()
    assert apply_n_action_steps(config, 25) == 25
    assert config.n_action_steps == 50


def test_replanner_is_due_after_every_n_actions():
    replanner = Replanner(3)
    assert [replanner.executed() for _ in range(7)] == [False, False, True, False, False, True, False]


def test_replanner_restarts_its_count_on_reset():
    replanner = Replanner(3)
    replanner.executed()
    replanner.executed()
    replanner.reset()
    assert [replanner.executed() for _ in range(3)] == [False, False, True]


def test_replanner_without_a_horizon_is_never_due():
    replanner = Replanner(None)
    assert not any(replanner.executed() for _ in range(100))
