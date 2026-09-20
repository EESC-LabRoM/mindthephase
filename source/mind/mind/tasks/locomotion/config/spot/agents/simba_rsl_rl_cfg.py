"""SimBa-based ActorCritic for Spot locomotion + its PPO runner config.

One-file scaffold: swap out the SimBa trunk below to iterate on architectures.

rsl_rl version contract
-----------------------
Targets the current rsl_rl (TensorDict-based API). Contract we rely on:

1. `OnPolicyRunner` instantiates the policy via
       eval(class_name)(obs, obs_groups, num_actions, **policy_cfg)
   where obs is a TensorDict and obs_groups maps "policy"/"critic" to lists
   of obs-group keys.
2. The base `ActorCritic` routes obs via `get_actor_obs` / `get_critic_obs`
   (concatenating the configured groups) before calling `self.actor(obs)`
   and `self.critic(obs)` on a flat (B, D) tensor -- so swapping the two
   trunks is still sufficient.
3. Gaussian distribution (`Normal(mean, std)`) is set up by the base
   `__init__` and reused by our inherited `act` / `get_actions_log_prob` /
   `action_mean` / `action_std` / `entropy`. Different dist => override.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from rsl_rl.modules.actor_critic import ActorCritic

from mind.tasks.locomotion.locomotion_env_cfg import (
    left_right_symmetry_obs,
)

import torch
import torch.nn as nn

import os
from datetime import datetime
from dataclasses import field
from dotenv import load_dotenv

load_dotenv()


class _PreLNResidualBlock(nn.Module):
    """Pre-LN residual block: x + Linear2(ReLU(Linear1(LayerNorm(x))))."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(torch.relu(self.linear1(self.norm(x))))


class _SimbaTrunk(nn.Module):
    """SimBa trunk: embedder -> N x PreLNResidualBlock -> post-LN -> output head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        final_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[_PreLNResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.zeros_(self.embed.bias)
        nn.init.orthogonal_(self.head.weight, gain=final_gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.post_ln(self.blocks(self.embed(x))))

    def __getitem__(self, idx):
        return self.embed  # exporter uses [0].in_features to size the dummy ONNX input


class SimbaActorCritic(ActorCritic):
    """rsl_rl ActorCritic with `self.actor`/`self.critic` replaced by SimBa trunks.

    All PPO plumbing (distribution, act, evaluate, log_prob, entropy, ...) is
    inherited from the base class, which only interacts with the networks
    through `self.actor(obs) -> (B, num_actions)` and
    `self.critic(obs) -> (B, 1)`.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "relu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_hidden_dim: int = 256,
        actor_num_blocks: int = 1,
        critic_hidden_dim: int = 512,
        critic_num_blocks: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        # Compute flat input dims from obs groups (mirrors base class logic).
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        # Replace the MLP actor/critic built by super().__init__ with SimBa trunks.
        self.actor = _SimbaTrunk(num_actor_obs, num_actions, actor_hidden_dim, actor_num_blocks, final_gain=0.01)
        self.critic = _SimbaTrunk(num_critic_obs, 1, critic_hidden_dim, critic_num_blocks, final_gain=1.0)
        print(f"SimBa Actor: {self.actor}")
        print(f"SimBa Critic: {self.critic}")


@configclass
class SpotLocomotionSimbaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the PPO runner."""

    num_steps_per_env = 24
    max_iterations = 15_000
    save_interval = 200
    experiment_name = "mind" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name: str = field(default_factory=lambda: f"mind_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger = "wandb"
    wandb_entity = "tommaselli"
    wandb_project = "mymind"
    empirical_normalization = True  # surrogate for the first RMSNorm
    policy = RslRlPpoActorCriticCfg(
        class_name=(
            "__import__('mind.tasks.locomotion.config.spot.agents.simba_rsl_rl_cfg',"
            " fromlist=['SimbaActorCritic']).SimbaActorCritic"
        ),
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="relu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=left_right_symmetry_obs,
        ),
    )
