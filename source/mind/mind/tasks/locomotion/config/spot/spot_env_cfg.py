from __future__ import annotations

from isaaclab.utils import configclass

from mind.tasks.locomotion.locomotion_env_cfg import LocomotionEnvCfg

from mind.assets.spot.spot import SPOT_ARM_CFG  # isort: skip

from isaaclab.envs import ViewerCfg


@configclass
class SpotLocomotionEnvCfg(LocomotionEnvCfg):
    """Configuration for the Spot locomotion environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None


@configclass
class SpotPlayEnvCfg(SpotLocomotionEnvCfg):
    """Configuration for the Spot locomotion environment."""

    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        eye=(3.0, 3.0, 3.0),
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 0.50

        # Only forward
        self.commands.base_velocity.ranges.lin_vel_y = (0.50, 0.50)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_orientation.flat_envs = 1.0
        self.commands.base_height.height_range = (0.6, 0.6)

        # No noise
        self.observations.policy.enable_corruption = False

        # flat terraing
        self.scene.terrain.terrain_type = "plane"

        # Remove pushs
        self.events.push_robot = None
        self.events.external_force_torque = None
        self.events.add_leg_mass = None

        # debug view command
        self.commands.base_velocity.debug_vis = True
        self.curriculum = None

        # Deactivate reset randomization
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
