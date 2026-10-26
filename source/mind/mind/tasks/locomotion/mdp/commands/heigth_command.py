# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


class UniformHeightCommand(CommandTerm):
    """Command term that samples a uniform height for the robot torso."""

    cfg: UniformHeightCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

        self.asset: RigidObject = env.scene[cfg.asset_name]

        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tHeight range: {self.cfg.height_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired torso height. Shape is (num_envs, 1)."""
        return self.height_command

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        self.height_command[env_ids] = sample_uniform(
            self.cfg.height_range[0],
            self.cfg.height_range[1],
            (len(env_ids), 1),
            device=self.device,
        )

    def _update_command(self):
        """Update the command (if needed, e.g. for interpolation)."""

    def _update_metrics(self):
        """Update the metrics for the command term."""
        self.metrics["height_error"] = torch.abs(
            self.asset.data.root_pos_w[:, 2] - self.height_command.squeeze(-1)
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up visualization markers for debugging."""

    def _debug_vis_callback(self, event):
        """Update visualization markers each step."""


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for a uniform height command term."""

    class_type: type = UniformHeightCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    height_range: tuple[float, float] = MISSING
    """Range for sampling height [min, max]."""