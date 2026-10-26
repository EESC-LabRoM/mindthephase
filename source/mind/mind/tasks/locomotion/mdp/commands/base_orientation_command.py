from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, sample_uniform


class UniformBaseOrientationCommand(CommandTerm):
    """Command term that samples a uniform base orientation (roll, pitch) for the robot.

    Note: Yaw is omitted from this command as it is typically controlled via velocity/heading commands.
    """

    cfg: UniformBaseOrientationCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: UniformBaseOrientationCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.orientation_command = torch.zeros(self.num_envs, 2, device=self.device)

        self.orientation_command_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.orientation_command_quat[:, 0] = 1.0

        self._yaw_zeros = torch.zeros(self.num_envs, device=self.device)

        self._low = torch.tensor(
            [cfg.roll_range[0], cfg.pitch_range[0]],
            device=self.device,
        )
        self._high = torch.tensor(
            [cfg.roll_range[1], cfg.pitch_range[1]],
            device=self.device,
        )

        self.is_flat_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        num_flat_envs = int(self.num_envs * cfg.flat_envs)
        self.is_flat_env[:num_flat_envs] = True

        self._marker_pos_offset = torch.tensor(cfg.marker_pos_offset, device=self.device)

        self.asset: RigidObject = env.scene[cfg.asset_name]

        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformBaseOrientationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tRoll range: {self.cfg.roll_range}\n"
        msg += f"\tPitch range: {self.cfg.pitch_range}\n"
        msg += f"\tFlat envs fraction: {self.cfg.flat_envs}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base orientation (roll, pitch). Shape is (num_envs, 2)."""
        return self.orientation_command

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        self.orientation_command[env_ids] = sample_uniform(
            self._low,
            self._high,
            (len(env_ids), 2),
            device=self.device,
        )

        self.orientation_command[env_ids] = torch.where(
            self.is_flat_env[env_ids].unsqueeze(-1),
            torch.zeros_like(self.orientation_command[env_ids]),
            self.orientation_command[env_ids],
        )

        self.orientation_command_quat[env_ids] = quat_from_euler_xyz(
            self.orientation_command[env_ids, 0],
            self.orientation_command[env_ids, 1],
            self._yaw_zeros[env_ids],
        )

    def _update_command(self):
        """Update the command (if needed, e.g. for interpolation)."""

    def _update_metrics(self):
        """Update the metrics for the command term."""
        roll, pitch, _ = euler_xyz_from_quat(self.asset.data.root_quat_w)
        current_roll_pitch = torch.stack((roll, pitch), dim=-1)

        self.metrics["orientation_error"] = torch.linalg.norm(
            self.orientation_command - current_roll_pitch,
            dim=-1,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up visualization markers for debugging."""
        if debug_vis:
            if not hasattr(self, "visualizer"):
                self.visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.visualizer.set_visibility(True)
        else:
            if hasattr(self, "visualizer"):
                self.visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update visualization markers each step."""
        _, _, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)

        marker_quat = quat_from_euler_xyz(
            self.orientation_command[:, 0],
            self.orientation_command[:, 1],
            yaw,
        )

        marker_pos = self.asset.data.root_pos_w + self._marker_pos_offset
        self.visualizer.visualize(translations=marker_pos, orientations=marker_quat)


@configclass
class UniformBaseOrientationCommandCfg(CommandTermCfg):
    """Configuration for a uniform base orientation command term."""

    class_type: type = UniformBaseOrientationCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    flat_envs: float = 0.2
    """Fraction of environments that are flat."""

    roll_range: tuple[float, float] = MISSING
    """Range for sampling roll [min, max] in radians."""

    pitch_range: tuple[float, float] = MISSING
    """Range for sampling pitch [min, max] in radians."""

    visualizer_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="/Visuals/Command/base_orientation_goal",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.5, 0.3, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    opacity=0.5,
                ),
            )
        },
    )
    """Configuration for the visualization markers."""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.4)
    """Position offset for the visualization marker (w.r.t. asset root position)."""
