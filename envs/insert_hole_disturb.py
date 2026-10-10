"""insert_hole with a disturbance inside the task: the slot moves during the play phase.

Scene, grasp and pre-move are insert_hole's on the same seed. After the pre-move, with probability
``shift_prob``, the (kinematic) slot moves horizontally by ``shift_distance`` metres in a random direction,
starting at a random physics step in ``shift_window`` (counted from the end of the pre-move, 120 Hz, so
collection and evaluation see the same simulated time) at ``shift_speed`` metres per physics step; the
schedule has its own generator (``envs.utils.target_shift``) and never changes the scene. The disturbance is
part of the task: it runs in collection and in evaluation alike.

Two experts, both insert_hole's own moves:
  non-reactive (``correct_on_trigger`` false): the original three moves, computed once; the slot's motion
    is ignored (it is what the open-loop expert would do).
  reactive (``monitor_push``, ``contact_trigger_px``, ``correct_on_trigger``): the push stops when the marker
    field moves, then the prism is re-aligned from the privileged pose of the slot as it is now, and pushed.
The success check uses the hole frame of the slot's current pose.
"""

from __future__ import annotations

import numpy as np

from .insert_hole import Task as InsertHoleTask
from .insert_hole import TaskCfg as InsertHoleTaskCfg
from ._base_task import configclass
from .utils.target_shift import draw_schedule, schedule_rng
from .utils.transforms import Pose


@configclass
class TaskCfg(InsertHoleTaskCfg):
    shift_prob: float = 1.0
    # Physics steps (120 Hz) after the pre-move. insert_hole's expert presses down for ~180 steps and aligns
    # for ~50, so 240-320 falls inside its insertion push.
    shift_window: tuple = (240, 320)
    shift_distance: tuple = (0.004, 0.004)
    # Metres per physics step: 0.25 mm = 30 mm/s.
    shift_speed: float = 0.00025


class Task(InsertHoleTask):
    def _reset_actors(self):
        super()._reset_actors()
        self.shift = draw_schedule(
            schedule_rng(self.cfg.seed),
            self.cfg.shift_prob,
            tuple(self.cfg.shift_window),
            tuple(self.cfg.shift_distance),
            self.cfg.shift_speed,
        )
        self.metadata["shift"] = None if self.shift is None else self.shift.as_metadata()
        self._phase_start = None
        self._slot_origin = None

    # -- the disturbance ---------------------------------------------------------------------------------

    def _step(self, is_save: bool = True):
        if not self.in_pre_move and self.mode != "eval_test":
            if self._phase_start is None:
                self._phase_start = self._physics_step_count
                self._slot_origin = self.slot.get_pose()
            phase = self._physics_step_count - self._phase_start
            if self.shift is not None and phase >= self.shift.start:
                if phase == self.shift.start:
                    self.metadata["shift_atom"] = [int(self.atom_id), str(self.atom_tag), int(self.step_count)]
                offset = self.shift.offset(phase)
                self.slot.set_pose(self._slot_origin.add_bias(offset, coord="world"))
        return super()._step(is_save)

    def _hole_frames(self) -> tuple[Pose, Pose]:
        """``(hole_pose, target_pose)`` of the slot's current pose, as insert_hole's pre-move builds them."""
        base_pose = self.slot.get_pose()
        base_pose[3:] = (1, 0, 0, 0)
        hole_pose = base_pose.add_bias([0.0, 0, 0.1])
        tilt = -np.pi / 6 if self.rotate == 0 else np.pi / 6
        return hole_pose, hole_pose.add_rotation([0, tilt, 0])

    # -- the reactive expert's correction reads the slot where it is now ----------------------------------

    def _before_correction(self):
        self.hole_pose, self.target_pose = self._hole_frames()

    def check_success(self, z_threshold=0.04):
        self.hole_pose, self.target_pose = self._hole_frames()
        return super().check_success(z_threshold)
