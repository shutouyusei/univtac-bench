from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    # Reactive expert; the defaults are the original expert. The alignment uses the prism-to-target tilt plus an
    # error of uniform magnitude in [min, max] degrees with a random sign, drawn per seed from a generator of its
    # own so the scene's random draws are unchanged.
    tilt_error_deg: tuple = (0.0, 0.0)
    # Record the marker field's RMS change from the start of the insertion push at every push step.
    monitor_push: bool = False
    # With monitor_push: stop the push once the change exceeds this many pixels (0 = never stop).
    contact_trigger_px: float = 0.0
    # After a stop: back off retract metres along the prism axis, re-align from the privileged pose, and push
    # to the original depth.
    correct_on_trigger: bool = False
    retract: float = 0.01

class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        slot_pose = Pose([0.6, 0.0, 0.001], [1, 0, 0, 0])
        base_pose = Pose([0.4, 0.0, 0.001], [1, 0, 0, 0])
        prism_pose = Pose([0.4, 0.0, 0.003], [1, 0, 0, 0])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="TestTubeHoleSlot.usd",
            pose=slot_pose,
            motion_type="kinematic",
        )
        self.prism_base = self._actor_manager.add_from_usd_file(
            name='prism_base',
            asset_path="TestTubeBase.usd",
            pose=base_pose,
            motion_type="kinematic",
        )
        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="TestTube.usd",
            pose=prism_pose,
            density=10
        )
    
    def _reset_actors(self):
        self.rotate = self.rng.choice([0, np.pi])
        base_offset = self.create_noise([0.02, 0.03, 0.0]).add_rotation([0, 0, self.rotate])
        base_pose = Pose([0.6, 0.0, 0.002], [1, 0, 0, 0]).add_offset(base_offset)
        self.slot.set_pose(base_pose)
    
    def pre_move(self):
        self.delay(10)
        
        grasp_bias = self.rng.uniform(0.095, 0.10)
        target_pose = self.prism.get_pose().add_bias([0, 0, grasp_bias])

        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            [1, 0, 0]
        )
        self.cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism,
            contact_point_id=self.cid,
            pre_dis=0.0, dis=0.0
        ))
        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())

        base_pose = self.slot.get_pose()
        base_pose[3:] = (1, 0, 0, 0)
        self.metadata['rotate'] = self.rotate

        self.hole_pose = base_pose.add_bias([0.0, 0, 0.1])
        if self.rotate == 0:
            self.target_pose = self.hole_pose.add_rotation([0, -np.pi/6, 0])
        else:
            self.target_pose = self.hole_pose.add_rotation([0, np.pi/6, 0])
        try_pose = self.hole_pose

        self.move(self.atom.move_by_displacement(z=0.15), constraint_pose=[1, 1, 1, 1, 1, 0])
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=try_pose,
            pre_dis=0.05, dis=0.01,
            is_open=False
        ))
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=try_pose,
            pre_dis=0.01, dis=0.002,
            is_open=False
        ), constraint_pose=[1, 1, 1, 1, 1, 0])

        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())

    def _play_once(self):
        self.move(self.atom.move_by_displacement(z=-0.03), time_dilation_factor=0.2)
        error = np.deg2rad(self._tilt_error())
        self.metadata['tilt_error_deg'] = float(np.rad2deg(error))
        self._align(error)
        axis = -self.prism.get_pose().to_transformation_matrix()[:3, 2]
        start = np.asarray(self._robot_manager.get_gripper_center_pose()[:3], dtype=float)
        self._push(0.04, monitor=self.cfg.monitor_push)
        if self.dense_action_stopped and self.cfg.correct_on_trigger:
            pushed = float(np.dot(np.asarray(self._robot_manager.get_gripper_center_pose()[:3]) - start, axis))
            self.metadata['stopped_after_m'] = pushed
            self.move(self.atom.move_by_displacement(
                z=self.cfg.retract, xyz_coord=self.prism.get_pose()
            ), time_dilation_factor=0.5)
            self._align(0.0)
            self._push(0.04 - pushed + self.cfg.retract, monitor=False)
            self.metadata['corrected'] = True
        self.delay(20, is_save=False)

    def _tilt_error(self) -> float:
        low, high = self.cfg.tilt_error_deg
        if high <= 0.0:
            return 0.0
        rng = np.random.default_rng([self.cfg.seed, 61])
        return float(rng.uniform(low, high) * rng.choice([-1.0, 1.0]))

    def _align(self, error: float):
        """The one alignment displacement, computed from the privileged prism-to-target tilt plus ``error`` (rad)."""
        rel_pose = self.prism.get_pose().rebase(self.target_pose)
        gripper_dis = self._robot_manager.get_gripper_center_pose().rebase(self.prism.get_pose())[2]
        tilt = rel_pose.euler[1] + error
        x_move = - gripper_dis * np.sin(tilt)
        z_move = gripper_dis * (np.cos(tilt) - 1)
        self.move(self.atom.move_by_displacement(
            x = x_move, z = z_move
        ), time_dilation_factor=0.5)

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

    def _push(self, distance: float, monitor: bool):
        """Push ``distance`` metres along the prism axis; with ``monitor`` record the marker field's RMS change from
        the start of the push and stop once it exceeds contact_trigger_px (if set)."""
        if monitor:
            reference = self._marker_field()
            trace = self.metadata.setdefault('push_marker_rms', [])

            def check():
                rms = float(np.sqrt(np.mean((self._marker_field() - reference) ** 2)))
                trace.append(rms)
                if 0.0 < self.cfg.contact_trigger_px < rms:
                    self.metadata['trigger_step'] = len(trace) - 1
                    return True
                return False

            self.dense_action_monitor = check
        try:
            self.move(self.atom.move_by_displacement(
                z=-distance, xyz_coord=self.prism.get_pose()
            ), time_dilation_factor=0.5)
        finally:
            self.dense_action_monitor = None
    
    def check_early_stop(self):
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        inhand_bias = np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2])
        if inhand_bias > 0.04:
            self.metadata['early_stop'] = True
            self.metadata['inhand_bias'] = float(inhand_bias)
            return True

    def check_success(self, z_threshold=0.04):
        prism_pose = self.prism.get_pose().rebase(self.target_pose)
        prism_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        self.metadata['rel_pose'] = prism_pose.tolist()
        return np.all(np.abs(prism_pose[:2]) < np.array([0.01, 0.01])) \
            and prism_pose[2] < -z_threshold \
            and np.dot(
                prism_pose.to_transformation_matrix()[:3, 2],
                [0, 0, 1]
            ) > 0.99 \
            and np.abs(self.origin_inhand_pose[2] - prism_inhand_pose[2]) < 0.04
