import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import tensordict
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from mind.assets.spot.constants import (
    ORDERED_JOINT_NAMES_SPOT_ARM,
    ORDERED_JOINT_NAMES_SPOT_BASE,
)
from mind.assets.spot.spot import SPOT_ARM_CFG
from mind.tasks.locomotion.mdp import locomotion_mdp_event as locomotion_mdp
from mind.tasks.locomotion.mdp import rewards as loco_mdp
from mind.tasks.locomotion.mdp.actions.fixed_joint_actions import (
    FixedJointPositionActionCfg,
)
from mind.tasks.locomotion.mdp.commands.base_orientation_command import (
    UniformBaseOrientationCommandCfg,
)
from mind.tasks.locomotion.mdp.commands.heigth_command import UniformHeightCommandCfg
from mind.tasks.locomotion.mdp.commands.uniform_vel_with_metrics import (
    UniformVelocityCommandWithMetricsCfg,
)
from mind.tasks.locomotion.mdp.observations import symmetry


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]),
            "threshold": 1.0,
        },
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DistantLightCfg(intensity=750.0),
    )


def undesired_contacts_squared(env, threshold, sensor_cfg):
    """Penalize undesired contacts using the sum of squared terms."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history

    squared_forces = torch.sum(torch.square(net_contact_forces[:, :, sensor_cfg.body_ids]), dim=-1)

    peak_squared_forces = torch.max(squared_forces, dim=1)[0]

    violation = peak_squared_forces - (threshold**2)

    return torch.sum(violation.clip(min=0.0), dim=1)


def base_linear_stoped_velocity_reward(env, asset_cfg, std: float) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    asset = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("base_velocity")
    lin_vel_error = torch.linalg.norm((asset.data.root_lin_vel_b), dim=1)
    is_zero = torch.linalg.norm(target, dim=1) == 0
    return torch.exp(-lin_vel_error / std) * is_zero


@configclass
class RewardsCfg:
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=10.0,
        params={
            "std": 2.0,
            "ramp_rate": 0.5,
            "ramp_at_vel": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    base_linear_velocity_2 = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=10.0,
        params={
            "std": 1.0,
            "ramp_rate": 0.5,
            "ramp_at_vel": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    should_stay_still = RewardTermCfg(
        func=base_linear_stoped_velocity_reward,
        weight=10.0,
        params={
            "std": 0.10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=10.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    base_angular_velocity_finer2 = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )

    foot_clearance = RewardTermCfg(
        func=loco_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    feet_stance_reward = RewardTermCfg(
        func=loco_mdp.feet_stance_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    gait = RewardTermCfg(
        func=loco_mdp.GaitReward,
        weight=5.0,
        params={
            "std": 0.025,
            "max_err": 0.5,
            "velocity_threshold": 0.1,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    foot_impact = RewardTermCfg(
        func=loco_mdp.foot_impact_penalty_2,
        weight=-0.00,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.000)

    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=loco_mdp.track_base_projected_gravity_l2,
        weight=-3.0,
        params={"command_name": "base_orientation", "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.70,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ORDERED_JOINT_NAMES_SPOT_BASE),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.1,
        },
    )

    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ORDERED_JOINT_NAMES_SPOT_BASE)},
    )

    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )

    contact_forces = RewardTermCfg(
        func=undesired_contacts_squared,
        weight=-2e-7,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 200.0,
        },
    )


def time_(env):
    sim_time: float = env._sim_step_counter * env.physics_dt
    sim_tensor = torch.tensor([sim_time], device=env.device).repeat(env.num_envs, 1)
    return sim_tensor


def last_action(
    env,
    action_name: str | None = None,
    scale: float = 1,
    offset: float = 0,
) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return (env.action_manager.action - offset) / scale
    else:
        return (env.action_manager.get_term(action_name).raw_actions - offset) / scale


def joint_pos_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    is_relative: bool = True,
):
    """The joint positions of the asset w.r.t. the default joint positions.

    Only joints configured in :attr:`asset_cfg.joint_ids` are returned.
    """
    asset = env.scene[asset_cfg.name]
    if is_relative:
        return (
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        )
    else:
        return asset.data.joint_pos[:, asset_cfg.joint_ids]


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class SpotObs(ObsGroup):
        """Observations for spot group."""

        sim_time = ObsTerm(func=time_)

        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        root_ang_vel_w = ObsTerm(
            func=mdp.root_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE + ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE + ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
        )

        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE + ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        actions = ObsTerm(func=last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_pos = ObsTerm(
            func=joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE + ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE + ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )

        actions = ObsTerm(func=last_action, params={"action_name": "joint_pos"})
        height_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_height"},
        )

        base_orientation_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_orientation"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    spot: SpotObs = SpotObs()


N_LEG = len(ORDERED_JOINT_NAMES_SPOT_BASE)
N_ARM = len(ORDERED_JOINT_NAMES_SPOT_ARM)

OBS_DIMS = [
    (3, symmetry.left_right_symmetry_base_lin_vel),
    (3, symmetry.left_right_symmetry_base_ang_vel),
    (3, symmetry.left_right_symmetry_projected_gravity),
    (3, symmetry.left_right_symmetry_velocity_commands),
    (N_LEG, symmetry.left_right_symmetry_base_joint_pos),
    (N_ARM, symmetry.left_right_symmetry_arm_joints),
    (N_LEG, symmetry.left_right_symmetry_base_joint_vel),
    (N_ARM, symmetry.left_right_symmetry_arm_joint_vel),
    (N_LEG, symmetry.left_right_symmetry_actions),
    (1, symmetry.left_right_symmetry_height_commands),
    (2, symmetry.left_right_symmetry_base_orientation_commands),
]
TOTAL_DIM = sum(dim for dim, _ in OBS_DIMS)


def left_right_proprio_symmetry_obs(
    obs: torch.Tensor,
) -> torch.Tensor:
    last = 0
    for dim, func in OBS_DIMS:
        obs[..., last : last + dim] = func(obs[..., last : last + dim])
        last += dim

    if last != obs.shape[-1]:
        raise ValueError(f"Observation dimension mismatch: expected {last}, got {obs.shape[-1]}")

    return obs


def left_right_symmetry_obs(
    env,
    obs: tensordict.TensorDict,
    actions: torch.Tensor,
):
    """Args:

        obs: Observation tensor dictionary, or None to omit observation augmentation.
        actions: Action tensor, or None to omit action augmentation.

    Returns:
        Augmented observation dictionary and action tensors, with omitted inputs
        returned as None.
    """
    if obs is not None:
        obs_clone = obs.clone()
        obs_clone["policy"] = left_right_proprio_symmetry_obs(obs_clone["policy"])
        obs = torch.cat([obs, obs_clone], dim=0)

    if actions is not None:
        actions_ = symmetry.left_right_symmetry_actions(actions.clone())
        actions = torch.cat([actions, actions_], dim=0)

    return obs, actions


def replace_value_at_step(env, env_ids, data, value, num_steps):
    """Helper function to replace a value after a given number of steps.

    Args:
        env: The learning environment
        env_ids: Environment IDs
        data: Current value
        value: New value to set
        num_steps: Number of steps after which to apply the new value

    Returns:
        New value if step threshold is reached, otherwise NO_CHANGE
    """
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_env_param.NO_CHANGE


@configclass
class CurriculumsCfg:
    """Curriculum configuration for gradually enabling perturbations."""

    enable_external_force_torque_force = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.external_force_torque.params.force_range",
            "modify_fn": replace_value_at_step,
            "modify_params": {
                "value": (-10.0, 10.0),
                "num_steps": 4000,
            },
        },
    )

    enable_external_force_torque_torque = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.external_force_torque.params.torque_range",
            "modify_fn": replace_value_at_step,
            "modify_params": {
                "value": (-2.0, 2.0),
                "num_steps": 7000,
            },
        },
    )

    enable_push_robot_vel = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.push_robot.params.velocity_range",
            "modify_fn": replace_value_at_step,
            "modify_params": {
                "value": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
                "num_steps": 10000,
            },
        },
    )


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    add_arm_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm_link_sh0"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*leg"),
            "mass_distribution_params": (0.8, 1.5),
            "operation": "scale",
        },
    )

    external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (
                0.0,
                0.0,
            ),
            "torque_range": (
                0.0,
                0.0,
            ),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (-0.50, 0.50),
                "y": (-0.50, 0.50),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=locomotion_mdp.reset_base_and_arm_joints_stow,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "arm_joint_names": ORDERED_JOINT_NAMES_SPOT_ARM,
            "base_joint_names": ORDERED_JOINT_NAMES_SPOT_BASE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(7.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (
                    0.0,
                    0.0,
                ),
                "pitch": (
                    0.0,
                    0.0,
                ),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = UniformVelocityCommandWithMetricsCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.5,
        heading_command=True,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )

    base_orientation = UniformBaseOrientationCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        flat_envs=0.4,
        roll_range=(-0.5, 0.5),
        pitch_range=(-0.5, 0.5),
    )

    base_height = UniformHeightCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), height_range=(0.5, 0.7)
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ORDERED_JOINT_NAMES_SPOT_BASE,
        scale=0.2,
        use_default_offset=True,
        preserve_order=True,
    )

    arm_joint_pos = FixedJointPositionActionCfg(
        asset_name="robot",
        joint_names=ORDERED_JOINT_NAMES_SPOT_ARM,
    )


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        self.scene.contact_forces.update_period = self.sim.dt
