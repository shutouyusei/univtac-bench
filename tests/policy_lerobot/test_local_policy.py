"""Inference-time overrides the server applies to a checkpoint's config before building the policy."""

from types import SimpleNamespace

import pytest

from policy.lerobot.bridge import local_policy


def test_steps_per_block_override_rewrites_the_step_count_too():
    config = SimpleNamespace(type="tacforcing", chunk_size=50, block_size=5, steps_per_block=1, num_steps=10)
    local_policy.apply_steps_per_block(config, 5)
    assert (config.steps_per_block, config.num_steps) == (5, 50)


def test_no_override_leaves_the_checkpoint_setting():
    config = SimpleNamespace(type="tacforcing", chunk_size=50, block_size=5, steps_per_block=2, num_steps=20)
    local_policy.apply_steps_per_block(config, None)
    assert (config.steps_per_block, config.num_steps) == (2, 20)


def test_the_override_is_refused_on_a_policy_without_blocks():
    config = SimpleNamespace(type="smolvla", chunk_size=50, num_steps=10)
    with pytest.raises(ValueError, match="steps_per_block"):
        local_policy.apply_steps_per_block(config, 5)


def test_the_override_must_be_a_positive_count():
    config = SimpleNamespace(type="tacforcing", chunk_size=50, block_size=5, steps_per_block=1, num_steps=10)
    with pytest.raises(ValueError):
        local_policy.apply_steps_per_block(config, 0)
