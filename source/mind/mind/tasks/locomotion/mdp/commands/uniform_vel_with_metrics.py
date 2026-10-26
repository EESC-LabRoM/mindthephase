from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


class UniformVelocityCommandWithMetrics(UniformVelocityCommand):
    """Uniform velocity command with metrics."""
    cfg: 'UniformVelocityCommandWithMetricsCfg'
    """Configuration for the command term."""

    def __init__(self, cfg: 'UniformVelocityCommandWithMetricsCfg', env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.metrics["joint_torque"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["contact_forces"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_slip"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_motion"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_smoothness"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gait"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()

        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        reward_manager = self._env.reward_manager

        term_to_metric_key = {
            "joint_torques": "joint_torque",
            "contact_forces": "contact_forces",
            "foot_slip": "foot_slip",
            "action_smoothness": "action_smoothness",
            "gait": "gait",
        }

        for term_name, metric_key in term_to_metric_key.items():
            if term_name in reward_manager._term_names:
                term_idx = reward_manager._term_names.index(term_name)
                weight = reward_manager._term_cfgs[term_idx].weight

                raw_reward = reward_manager._step_reward[:, term_idx]

                if abs(weight) > 1e-9:
                    unweighted_val = raw_reward / weight
                else:
                    unweighted_val = raw_reward

                self.metrics[metric_key] += unweighted_val / max_command_step
            else:
                print("Warning: {} not found in reward manager".format(term_name))


    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommandWithMetrics:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

@configclass
class UniformVelocityCommandWithMetricsCfg(UniformVelocityCommandCfg):
    """Configuration for the command term."""
    class_type: type = UniformVelocityCommandWithMetrics