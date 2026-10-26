"""Shared SimBa policy configuration for the RSL RL 3.x runner."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class SimbaPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = (
        "__import__('mind.rsl_rl.simba', fromlist=['SimbaActorCritic']).SimbaActorCritic"
    )
    init_noise_std: float = 1.0
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [128, 128, 128]
    critic_hidden_dims: list[int] = [128, 128, 128]
    activation: str = "relu"
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    simba_depth_scale: bool = True
