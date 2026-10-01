from types import SimpleNamespace

import pytest

from policy.lerobot.bridge.local_policy import apply_n_action_steps


def chunk_config(**kw):
    return SimpleNamespace(type="smolvla", chunk_size=50, n_action_steps=50, **kw)


def test_none_keeps_the_checkpoint_value():
    config = chunk_config()
    apply_n_action_steps(config, None)
    assert config.n_action_steps == 50


def test_sets_the_executed_actions_per_chunk():
    config = chunk_config()
    apply_n_action_steps(config, 25)
    assert config.n_action_steps == 25


@pytest.mark.parametrize("n", [0, 51])
def test_rejects_values_outside_the_chunk(n):
    with pytest.raises(ValueError):
        apply_n_action_steps(chunk_config(), n)


def test_rejects_block_streaming_policies():
    config = chunk_config(steps_per_block=5, block_size=5)
    with pytest.raises(ValueError):
        apply_n_action_steps(config, 25)
