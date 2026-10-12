"""RDP training at toy sizes on CPU: both stages run, write their checkpoints and the evaluation bundle, resume
after an interruption, and skip a finished stage."""

import json

import numpy as np
import pytest
import torch

from conftest import TOY, write_demo
from rdp_univtac import build, marker, prepare, train

@pytest.fixture
def setup(tmp_path):
    demos = tmp_path / "hdf5"
    demos.mkdir()
    for i, T in enumerate((30, 26)):
        write_demo(demos / f"{i}.hdf5", T, i)
    cache = tmp_path / "cache"
    prepare.build_cache(prepare.demo_paths(demos, 2), cameras=["head"], out=cache)
    pca = tmp_path / "pca.npz"
    prepare.fit_pooled_pca([prepare.demo_paths(demos, 2)]).save(pca)
    return cache, pca, tmp_path / "out"


def epochs(out, stage):
    rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    return [r["epoch"] for r in rows if r["stage"] == stage]


def test_both_stages_and_the_bundle(setup):
    cache, pca, out = setup
    cfg = build.load_config(override=TOY)
    train.train_at(cfg, cache, pca, out, "cpu")
    at = torch.load(out / "at_latest.pt", weights_only=False)
    assert at["done"] and at["epoch"] == 2
    bundle_path = train.train_ldp(cfg, cache, pca, out, "cpu")
    bundle = torch.load(bundle_path, weights_only=False)
    assert set(bundle) >= {"policy", "at", "normalizer", "pca_mean", "pca_components", "shape_meta", "cameras", "config"}
    assert bundle["cameras"] == ["cam_head"]
    assert "normalizer.params_dict.latent_action.scale" in bundle["policy"]
    assert np.allclose(bundle["pca_mean"], marker.MarkerPCA.load(pca).mean)
    assert epochs(out, "at") == [1, 2] and epochs(out, "ldp") == [1, 2]
    # finished stages are skipped
    train.train_at(cfg, cache, pca, out, "cpu")
    train.train_ldp(cfg, cache, pca, out, "cpu")
    assert epochs(out, "at") == [1, 2] and epochs(out, "ldp") == [1, 2]


def test_the_policy_stage_needs_a_finished_tokenizer(setup):
    cache, pca, out = setup
    with pytest.raises(FileNotFoundError):
        train.train_ldp(build.load_config(override=TOY), cache, pca, out, "cpu")


def test_resume_after_an_interruption(setup, monkeypatch):
    cache, pca, out = setup
    cfg = build.load_config(override=build.deep_update(TOY, {"at_training": {"num_epochs": 3}}))
    real_log = train.log

    def interrupt_at_epoch_2(o, row):
        if row["epoch"] == 2:
            raise KeyboardInterrupt
        real_log(o, row)

    monkeypatch.setattr(train, "log", interrupt_at_epoch_2)
    with pytest.raises(KeyboardInterrupt):
        train.train_at(cfg, cache, pca, out, "cpu")
    saved = torch.load(out / "at_latest.pt", weights_only=False)
    assert saved["epoch"] == 1 and not saved["done"]
    monkeypatch.setattr(train, "log", real_log)
    train.train_at(cfg, cache, pca, out, "cpu")
    assert epochs(out, "at") == [1, 2, 3]
    assert torch.load(out / "at_latest.pt", weights_only=False)["done"]


def test_a_different_config_is_refused_instead_of_reused(setup):
    cache, pca, out = setup
    train.train_at(build.load_config(override=TOY), cache, pca, out, "cpu")
    other = build.load_config(override=build.deep_update(TOY, {"at_training": {"num_epochs": 3}}))
    with pytest.raises(RuntimeError, match="different config"):
        train.train_at(other, cache, pca, out, "cpu")


def test_a_lost_bundle_is_rewritten_without_training(setup):
    cache, pca, out = setup
    cfg = build.load_config(override=TOY)
    train.train_at(cfg, cache, pca, out, "cpu")
    bundle = train.train_ldp(cfg, cache, pca, out, "cpu")
    bundle.unlink()
    assert train.train_ldp(cfg, cache, pca, out, "cpu").exists()
    assert epochs(out, "ldp") == [1, 2]
    assert torch.load(bundle, weights_only=False)["config_hash"] == train.config_hash(cfg)


def test_parse_set():
    assert train.parse_set(["ldp_training.num_epochs=3", "ldp.down_dims=[8, 16]"]) == {
        "ldp_training": {"num_epochs": 3}, "ldp": {"down_dims": [8, 16]}}
