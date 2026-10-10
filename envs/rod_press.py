"""Rod pressing probe (stage 0 of the rod-press task family).

The gripper holds a test tube vertically, presses its tip on a kinematic board and then holds still while the
board moves up and down by a few millimetres. Every control step of the hold is logged to the metadata: the
board offset, the contact wrench on each gel pad, the marker-field change, the press depth and the rod's in-hand
slip. The probe asks one question before any task is built on it: does a millimetre-scale board motion show in
the fingertip tactile images and the contact force clearly above the standing noise, with a sign that follows
the direction of the motion?

No expert and no success band yet: the scene is fixed (no position noise) and the episode always counts as a
success so that the recording is kept.
"""

from ._base_task import *
import numpy as np


@configclass
class TaskCfg(BaseTaskCfg):
    # Board displacements (metres, world z, relative to the resting pose) visited in order after the press.
    probe_offsets: tuple = (0.0, 0.001, 0.002, 0.003, 0.0, -0.001, -0.002, -0.003, 0.0)
    # Control steps the board is held at each offset once it has arrived there.
    hold_steps: int = 40
    # Metres the board moves per control step on its way to the next offset (0.0002 = 24 mm/s at 120 Hz).
    ramp_per_step: float = 0.0002
    # Gap left between the rod tip and the board top before the press, and the commanded press past that point.
    approach_gap: float = 0.005
    press_depth: float = 0.002


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        # Rod and stand are insert_hole's test tube on its base; the board is a 10 cm kinematic cube whose top
        # face (z = 0.101) is the pressing surface.
        self.rod_base = self._actor_manager.add_from_usd_file(
            name='rod_base',
            asset_path="TestTubeBase.usd",
            pose=Pose([0.4, 0.0, 0.001], [1, 0, 0, 0]),
            motion_type="kinematic",
        )
        self.rod = self._actor_manager.add_from_usd_file(
            name='rod',
            asset_path="TestTube.usd",
            pose=Pose([0.4, 0.0, 0.003], [1, 0, 0, 0]),
            density=10,
        )
        self.board_origin = Pose([0.6, 0.0, 0.001], [1, 0, 0, 0])
        self.board = self._actor_manager.add_from_usd_file(
            name='board',
            asset_path="Stand.usd",
            pose=self.board_origin,
            motion_type="kinematic",
        )
        self.board_top = self.board_origin.p[2] + 0.1

    def _reset_actors(self):
        self.board.set_pose(self.board_origin)
        self.board_offset = 0.0

    def pre_move(self):
        self.delay(10)

        # insert_hole's grasp: 9.5-10 cm up the tube, fingers closing along world x.
        grasp_bias = self.rng.uniform(0.095, 0.10)
        target_pose = self.rod.get_pose().add_bias([0, 0, grasp_bias])
        cpose = construct_grasp_pose(target_pose.p, [0, 0, 1], [1, 0, 0])
        self.cid = self.rod.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(self.rod, contact_point_id=self.cid, pre_dis=0.0, dis=0.0))

        self.move(self.atom.move_by_displacement(z=0.15), constraint_pose=[1, 1, 1, 1, 1, 0])

        # Carry the rod over the board and lower its tip to approach_gap above the top face. The tube's origin
        # is its bottom face, so the tip height is the rod pose's z.
        above = Pose([self.board_origin.p[0], self.board_origin.p[1], self.board_top + 0.03], [1, 0, 0, 0])
        self.move(self.atom.place_actor(self.rod, target_pose=above, pre_dis=0.05, dis=0.0, is_open=False))
        gap = self.rod.get_pose().p[2] - (self.board_top + self.cfg.approach_gap)
        self.move(self.atom.move_by_displacement(z=-float(gap)), time_dilation_factor=0.5)

        self.origin_inhand_pose = self.rod.get_pose().rebase(self._robot_manager.get_gripper_center_pose())
        self.metadata['grasp_bias'] = float(grasp_bias)

    def _play_once(self):
        # The press: command the gripper past the contact point by press_depth; the rod cannot follow, so the
        # gels take the load.
        self.move(self.atom.move_by_displacement(z=-(self.cfg.approach_gap + self.cfg.press_depth)),
                  time_dilation_factor=0.2)
        self.marker_reference = self._marker_field()
        self.metadata['probe'] = []
        for target in self.cfg.probe_offsets:
            self._hold_at(float(target))
        self.delay(10, is_save=False)

    # -- the board ------------------------------------------------------------------------------------------

    def _hold_at(self, target: float):
        """Ramp the board to ``target`` at ramp_per_step, then hold it there for hold_steps steps; one record per
        control step throughout."""
        while True:
            remaining = target - self.board_offset
            if abs(remaining) <= 1e-9:
                break
            step = np.clip(remaining, -self.cfg.ramp_per_step, self.cfg.ramp_per_step)
            self._set_board(self.board_offset + step)
            self._step(is_save=True)
            self._record(target, settled=False)
        for _ in range(self.cfg.hold_steps):
            self._step(is_save=True)
            self._record(target, settled=True)

    def _set_board(self, offset: float):
        self.board_offset = float(offset)
        self.board.set_pose(self.board_origin.add_bias([0.0, 0.0, self.board_offset], coord='world'))

    # -- the readings ---------------------------------------------------------------------------------------

    def _marker_field(self) -> np.ndarray:
        """Marker displacement (current minus reference) of both fingertips in a fixed marker order."""
        self._update_render()
        fields = []
        for obs in self._tactile_manager.get_observations(['marker']).values():
            m = obs['marker'].detach().cpu().numpy()
            ref = np.round(m[0]).astype(int)
            order = np.lexsort((ref[:, 0], ref[:, 1]))
            fields.append((m[1] - m[0])[order].ravel())
        return np.concatenate(fields)

    def _record(self, target: float, settled: bool):
        self._update_render()
        tactile = self._tactile_manager.get_observations(['force', 'torque', 'press_depth'])
        field = self._marker_field() - self.marker_reference
        inhand = self.rod.get_pose().rebase(self._robot_manager.get_gripper_center_pose())
        rec = {
            'step': int(self.step_count),
            'target': target,
            'offset': self.board_offset,
            'settled': settled,
            'marker_rms_px': float(np.sqrt(np.mean(field ** 2))),
            'marker_mean_px': [float(v) for v in field.reshape(-1, 2).mean(axis=0)],
            'inhand_slip_m': float(inhand[2] - self.origin_inhand_pose[2]),
            'rod_tip_z': float(self.rod.get_pose().p[2]),
            'ee_z': float(self._robot_manager.get_gripper_center_pose().p[2]),
        }
        for name, obs in tactile.items():
            rec[f'{name}_force_N'] = [float(v) for v in obs['force']]
            rec[f'{name}_torque_Nm'] = [float(v) for v in obs['torque']]
            rec[f'{name}_press_depth_mm'] = float(obs['press_depth'].max())
        self.metadata['probe'].append(rec)

    def check_success(self):
        return True
