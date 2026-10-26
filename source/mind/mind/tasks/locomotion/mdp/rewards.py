# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse


def track_base_projected_gravity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-tracking of commanded base tilt using the projected gravity vector.

    This computes the L2-difference between the target projected gravity vector
    (derived from the base orientation command) and the current base projected gravity.
    Note that this naturally ignores heading (yaw) alignment and focuses purely on base tilt.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    torso_projected_gravity_goal = quat_apply_inverse(
        command_term.orientation_command_quat,
        asset.data.GRAVITY_VEC_W,
    )

    return torch.sum(
        torch.square(torso_projected_gravity_goal[:, :2] - asset.data.projected_gravity_b[:, :2]),
        dim=1,
    )


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]

    base_velocity_command = env.command_manager.get_command("base_velocity")
    command_magnitude = torch.norm(base_velocity_command, dim=1)

    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std)

    return torch.where(command_magnitude > 0.05, reward, 0.0)


def feet_stance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward all feet in contact when command is zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    base_velocity_command = env.command_manager.get_command("base_velocity")
    command_magnitude = torch.norm(base_velocity_command, dim=1)

    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    reward = torch.sum(is_contact.float(), dim=1)

    return torch.where(command_magnitude < 0.05, reward, 0.0)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing.
    Note that this reward is only for quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError(
                "This reward only supports gaits with two pairs of synchronized feet, "
                "like trotting."
            )
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sync_reward_0 = self._sync_reward_func(
            self.synced_feet_pairs[0][0],
            self.synced_feet_pairs[0][1],
        )
        sync_reward_1 = self._sync_reward_func(
            self.synced_feet_pairs[1][0],
            self.synced_feet_pairs[1][1],
        )
        sync_reward = sync_reward_0 * sync_reward_1
        async_reward_0 = self._async_reward_func(
            self.synced_feet_pairs[0][0],
            self.synced_feet_pairs[1][0],
        )
        async_reward_1 = self._async_reward_func(
            self.synced_feet_pairs[0][1],
            self.synced_feet_pairs[1][1],
        )
        async_reward_2 = self._async_reward_func(
            self.synced_feet_pairs[0][0],
            self.synced_feet_pairs[1][1],
        )
        async_reward_3 = self._async_reward_func(
            self.synced_feet_pairs[1][0],
            self.synced_feet_pairs[0][1],
        )
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        return torch.where(cmd > 0.0, sync_reward * async_reward, 0.0)

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(
            torch.square(air_time[:, foot_0] - air_time[:, foot_1]),
            max=self.max_err**2,
        )
        se_contact = torch.clip(
            torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clip(
            torch.square(air_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        se_act_1 = torch.clip(
            torch.square(contact_time[:, foot_0] - air_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_impact_penalty_2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize foot impact when coming into contact with the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_down_velocity = torch.clamp(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], max=0.0)
    reward = first_contact * torch.square(foot_down_velocity)
    return torch.sum(reward, dim=1)
