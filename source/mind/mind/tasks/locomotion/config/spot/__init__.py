"""uv run scripts/rsl_rl/play.py --task Isaac-Spot-Locomotion-Play-v0  --num_envs 2 --video"""

import gymnasium as gym

from . import agents, spot_env_cfg


gym.register(
    id="Isaac-Spot-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SpotLocomotionPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Spot-Locomotion-Simba-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.simba_rsl_rl_cfg.SpotLocomotionSimbaPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Spot-Locomotion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotPlayEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SpotLocomotionPPORunnerCfg,
    },
)
