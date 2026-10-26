"""SimBa variants of IsaacLab's ANYmal D and H1 rough locomotion tasks."""

import gymnasium as gym

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import (
    AnymalDRoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
    H1RoughPPORunnerCfg,
)

from mind.rsl_rl.simba_cfg import SimbaPolicyCfg


@configclass
class AnymalDRoughSimbaRunnerCfg(AnymalDRoughPPORunnerCfg):
    """ANYmal D rough PPO settings with the fork's SimBa actor and critic."""

    experiment_name = "anymal_d_rough_simba"
    policy: SimbaPolicyCfg = SimbaPolicyCfg()


@configclass
class H1RoughSimbaRunnerCfg(H1RoughPPORunnerCfg):
    """H1 rough PPO settings with the same SimBa actor and critic."""

    experiment_name = "h1_rough_simba"
    policy: SimbaPolicyCfg = SimbaPolicyCfg()


gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-Simba-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d."
            "rough_env_cfg:AnymalDRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": AnymalDRoughSimbaRunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-H1-Simba-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg:H1RoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": H1RoughSimbaRunnerCfg,
    },
)
