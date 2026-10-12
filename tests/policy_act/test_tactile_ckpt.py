"""UniVTAC-ACT's tactile backbone: the pretrained-encoder path resolves against policy/ACT whatever the cwd,
and a configured path that does not exist is an error instead of a silent random initialisation."""

import os
import sys
from pathlib import Path

import pytest

ACT_DIR = Path(__file__).resolve().parents[2] / "policy" / "ACT"
sys.path.insert(0, str(ACT_DIR / "detr"))

from models.backbone import TactileBackbone, resolve_tactile_ckpt  # noqa: E402
from models.position_encoding import PositionEmbeddingSine  # noqa: E402

RELEASED = ACT_DIR / "../../data/checkpoints/encoder.pth"


def build(ckpt):
    return TactileBackbone("resnet18", ckpt, ["tac_left", "tac_right"], True, False, PositionEmbeddingSine(256))


def test_relative_path_resolves_against_the_act_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_tactile_ckpt("../../data/checkpoints/encoder.pth") == (
        ACT_DIR / "../../data/checkpoints/encoder.pth").resolve()


def test_absolute_path_is_kept(tmp_path):
    p = tmp_path / "enc.pth"
    assert resolve_tactile_ckpt(str(p)) == p


@pytest.mark.parametrize("ckpt", [None, ""])
def test_no_path_means_random_init(ckpt):
    assert resolve_tactile_ckpt(ckpt) is None
    build(ckpt)


def test_missing_configured_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(str(tmp_path / "nowhere.pth"))


@pytest.mark.skipif(not RELEASED.exists(), reason="released encoder.pth not downloaded")
def test_released_encoder_loads_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import torch

    model = build("../../data/checkpoints/encoder.pth")
    ref = torch.load(RELEASED, weights_only=True)
    key = next(k for k in ref if k.startswith("backbone.") and ref[k].dtype.is_floating_point)
    got = dict(model.backbone.named_parameters())[key[len("backbone."):]]
    assert torch.equal(got.detach().cpu(), ref[key].cpu())


def test_act_configs_point_at_the_released_encoder():
    import yaml

    for name in ("train_config.yml", "train_config_all.yml", "train_config_smoke.yml"):
        cfg = yaml.safe_load(open(ACT_DIR / name))
        assert cfg["tactile_names"], name
        assert cfg["tactile_ckpt"] == "../../data/checkpoints/encoder.pth", name
    assert os.path.basename(str(RELEASED)) == "encoder.pth"
