"""Command line of the converter: how the auxiliary target options become ConvertOptions."""

from policy.lerobot import process_data
from policy.lerobot.convert.aux_targets import VirtualTargetParams

BASE = ["insert_hole", "clean51_force", "100"]


def test_the_default_conversion_has_no_auxiliary_target():
    args = process_data.parse_args(BASE)
    assert process_data.build_options(args).virtual_target is None
    assert process_data.default_repo_id(args) == "local/insert_hole-clean51_force-100"


def test_virtual_target_takes_the_papers_constants_on_forces_in_newtons():
    args = process_data.parse_args(BASE + ["--aux-target", "virtual_target"])
    assert process_data.build_options(args).virtual_target == VirtualTargetParams()
    assert process_data.build_options(args).virtual_target.force_scale == 1.0


def test_a_dataset_recorded_in_another_unit_can_be_converted_with_a_force_scale():
    args = process_data.parse_args(BASE + ["--aux-target", "virtual_target", "--force-scale", "14400"])
    assert process_data.build_options(args).virtual_target == VirtualTargetParams(force_scale=14400.0)


def test_the_constants_can_be_overridden():
    args = process_data.parse_args(
        BASE + ["--aux-target", "virtual_target", "--force-scale", "2", "--f-min", "1", "--f-max", "4",
                "--k-min", "100", "--k-max", "5000"]
    )
    assert process_data.build_options(args).virtual_target == VirtualTargetParams(
        force_scale=2.0, f_min=1.0, f_max=4.0, k_min=100.0, k_max=5000.0
    )


def test_a_dataset_with_auxiliary_targets_gets_its_own_name():
    args = process_data.parse_args(BASE + ["--aux-target", "virtual_target"])
    assert process_data.default_repo_id(args) == "local/insert_hole-clean51_force-100-vt"
