from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import omni.log
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class FixedJointPositionAction(ActionTerm):
    """Action term that holds joints at a fixed position matching their default configuration.

    This action term ignores any input actions (action_dim=0) and constantly commands
    the joints to their default positions as defined in the robot's data.
    """

    cfg: FixedJointPositionActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """Reference to the robot asset for accessing joint information and sending commands."""

    def __init__(self, cfg: FixedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._raw_actions = torch.empty(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.empty(self.num_envs, 0, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._offset[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        self._asset.set_joint_position_target(self._offset, joint_ids=self._joint_ids)


@configclass
class FixedJointPositionActionCfg(ActionTermCfg):
    """Configuration for fixed joint position action term.

    This action term holds the specified joints at their default (initial) position.
    It has an action dimension of 0, meaning it does not consume any actions from the agent.
    """

    class_type: type = FixedJointPositionAction
    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
