"""Spot flat-terrain tasks matching the paper's Spot-flat runs (W&B rank_v6).

Upstream ``Isaac-Velocity-Flat-Spot-v0`` draws half of its terrain tiles from a
2-5 cm random-rough sub-terrain. The paper's Spot-flat runs used plane tiles only,
so these tasks keep the upstream Spot environment and replace its terrain mix with
a single ``flat`` sub-terrain. Everything else (rewards, observations, events,
PPO settings) is the upstream Spot flat configuration.

The SimBa runner uses the paper's trunk: actor and critic both ``d = 256`` with the
same number of blocks (override ``agent.policy.{actor,critic}_num_blocks`` per size).
"""

import gymnasium as gym

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.agents.rsl_rl_ppo_cfg import (
    SpotFlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg

from mind.rsl_rl.simba_cfg import SimbaPolicyCfg


@configclass
class SpotPureFlatEnvCfg(SpotFlatEnvCfg):
    """Upstream Spot flat environment on plane tiles only."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = self.scene.terrain.terrain_generator.replace(
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        )


@configclass
class SpotFlatSimbaRunnerCfg(SpotFlatPPORunnerCfg):
    """Upstream Spot flat PPO settings with the paper's SimBa actor and critic."""

    experiment_name = "spot_flat_simba"
    policy: SimbaPolicyCfg = SimbaPolicyCfg(
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        actor_num_blocks=1,
        critic_num_blocks=1,
    )


gym.register(
    id="Isaac-Velocity-Flat-Spot-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SpotPureFlatEnvCfg,
        "rsl_rl_cfg_entry_point": SpotFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Spot-Simba-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SpotPureFlatEnvCfg,
        "rsl_rl_cfg_entry_point": SpotFlatSimbaRunnerCfg,
    },
)
