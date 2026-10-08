"""insert_hole with the hole shifted during the task, and an expert that tracks it.

Scene, grasp and pre-move are insert_hole's on the same seed. After the pre-move, with probability
``shift_prob``, the (kinematic) slot moves horizontally by ``shift_distance`` metres in a random direction,
starting at a random physics step in ``shift_window`` and at ``shift_speed`` metres per physics step (see
``envs.utils.target_shift``); the schedule has its own generator, so it never changes the scene. Time is
counted in physics steps (120 Hz) because a control step is one physics step in collection and two in
evaluation: the shift runs at the same simulated time in both.

The expert keeps insert_hole's three moves (press down, the alignment computed from the prism's pose in the
hole frame, insertion along the prism axis) but executes each one in short pieces, re-reading the slot's
ground-truth pose before every piece and adding the slot's displacement since the previous piece. It never
reads the tactile: the tactile shows the consequence of the shift (the wall pushing the prism in hand), and
the expert's correction follows it in time. The success check and the alignment use the hole frame of the
slot's current pose.
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
    shift_prob: float = 0.8
    # Physics steps (120 Hz) after the pre-move: 40-240 = 0.33-2 s.
    shift_window: tuple = (40, 240)
    shift_distance: tuple = (0.002, 0.004)
    # Metres per physics step: 0.25 mm = 30 mm/s.
    shift_speed: float = 0.00025
    # Pieces per expert move; the expert re-reads the slot before each piece.
    track_pieces: int = 6


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
        self._tracked_slot = None

    # -- the shift ---------------------------------------------------------------------------------------

    def _step(self, is_save: bool = True):
        if not self.in_pre_move and self.mode != "eval_test":
            if self._phase_start is None:
                self._phase_start = self._physics_step_count
                self._slot_origin = self.slot.get_pose()
            phase = self._physics_step_count - self._phase_start
            if self.shift is not None and phase >= self.shift.start:
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

    # -- the tracking expert ----------------------------------------------------------------------------

    def _slot_drift(self) -> np.ndarray:
        """World displacement of the slot since the last call (zero on the first call)."""
        now = np.asarray(self.slot.get_pose().p, dtype=float)
        drift = np.zeros(3) if self._tracked_slot is None else now - self._tracked_slot
        self._tracked_slot = now
        return drift

    def _tracked_move(self, vec, coord, time_dilation_factor: float, pieces: int | None = None):
        """``vec`` (in ``coord``: 'world' or a Pose re-evaluated each piece by the callable) split into
        pieces; before each piece the slot is read again and its displacement added in the world frame."""
        pieces = pieces or self.cfg.track_pieces
        vec = np.asarray(vec, dtype=float)
        for _ in range(pieces):
            frame = coord() if callable(coord) else coord
            target = self._robot_manager.get_ee_pose().add_bias(vec / pieces, coord=frame)
            target = target.add_bias(self._slot_drift(), coord="world")
            if not self.move(self.atom.move_to_pose(target), time_dilation_factor=time_dilation_factor, delay=False):
                return False
        return True

    def _play_once(self):
        self._slot_drift()
        self._tracked_move([0, 0, -0.03], "world", time_dilation_factor=0.2)
        self.hole_pose, self.target_pose = self._hole_frames()
        rel_pose = self.prism.get_pose().rebase(self.target_pose)
        gripper_dis = self._robot_manager.get_gripper_center_pose().rebase(self.prism.get_pose())[2]
        x_move = -gripper_dis * np.sin(rel_pose.euler[1])
        z_move = gripper_dis * (np.cos(rel_pose.euler[1]) - 1)
        self._tracked_move([x_move, 0, z_move], "world", time_dilation_factor=0.5)
        self._tracked_move([0, 0, -0.04], lambda: self.prism.get_pose(), time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self, z_threshold=0.04):
        self.hole_pose, self.target_pose = self._hole_frames()
        return super().check_success(z_threshold)
