from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

ARM_STOW_JOINT_POS: dict[str, float] = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.1415,
    "arm_el0": 3.1415,
    "arm_el1": 1.5655,
    "arm_wr0": 0.00,
    "arm_wr1": -1.5655,
    "arm_f1x": 0,
}


def reset_base_and_arm_joints_stow(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    arm_joint_names: list[str],
    base_joint_names: list[str],
    arm_position_noise: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset base joints around defaults and arm joints to stow position.

    Base joints are randomized uniformly within ``position_range`` / ``velocity_range``
    of their default values. Arm joints are set to ``ARM_STOW_JOINT_POS`` plus a small
    uniform noise, with zero velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    arm_joint_ids, _ = asset.find_joints(arm_joint_names, preserve_order=True)
    base_joint_ids, _ = asset.find_joints(base_joint_names, preserve_order=True)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    base_default_pos = asset.data.default_joint_pos[env_ids][:, base_joint_ids]
    base_default_vel = asset.data.default_joint_vel[env_ids][:, base_joint_ids]

    base_min_pos = base_default_pos + position_range[0]
    base_max_pos = base_default_pos + position_range[1]
    base_min_vel = base_default_vel + velocity_range[0]
    base_max_vel = base_default_vel + velocity_range[1]

    base_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, base_joint_ids]
    base_min_pos = torch.clamp(base_min_pos, min=base_pos_limits[..., 0], max=base_pos_limits[..., 1])
    base_max_pos = torch.clamp(base_max_pos, min=base_pos_limits[..., 0], max=base_pos_limits[..., 1])

    base_vel_limits = asset.data.soft_joint_vel_limits[env_ids][:, base_joint_ids]
    base_min_vel = torch.clamp(base_min_vel, min=-base_vel_limits, max=base_vel_limits)
    base_max_vel = torch.clamp(base_max_vel, min=-base_vel_limits, max=base_vel_limits)

    joint_pos[:, base_joint_ids] = sample_uniform(base_min_pos, base_max_pos, base_min_pos.shape, base_min_pos.device)
    joint_vel[:, base_joint_ids] = sample_uniform(base_min_vel, base_max_vel, base_min_vel.shape, base_min_vel.device)

    stow_vals = (
        torch.tensor(
            [ARM_STOW_JOINT_POS.get(name, 0.0) for name in arm_joint_names],
            dtype=torch.float32,
            device=asset.device,
        )
        .unsqueeze(0)
        .expand(len(env_ids), -1)
    )
    noise = sample_uniform(
        torch.full_like(stow_vals, arm_position_noise[0]),
        torch.full_like(stow_vals, arm_position_noise[1]),
        stow_vals.shape,
        stow_vals.device,
    )
    joint_pos[:, arm_joint_ids] = stow_vals + noise
    joint_vel[:, arm_joint_ids] = 0.0

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
