"""SimBa actor and critic for the RSL RL 3.x ActorCritic API.

The trunk follows the IsaacLab fork's pre-LayerNorm residual architecture,
including its 1/sqrt(number of blocks) branch scaling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic


class _PreLNResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, branch_scale: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.branch_scale = branch_scale
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch_scale * self.linear2(torch.relu(self.linear1(self.norm(x))))


class _SimbaTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        final_gain: float,
        depth_scale: bool,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_blocks < 0:
            raise ValueError("SimBa requires a positive hidden dimension and a nonnegative block count.")
        branch_scale = 1.0 / math.sqrt(num_blocks) if depth_scale and num_blocks else 1.0
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *(_PreLNResidualBlock(hidden_dim, branch_scale) for _ in range(num_blocks))
        )
        self.post_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.zeros_(self.embed.bias)
        nn.init.orthogonal_(self.head.weight, gain=final_gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.post_ln(self.blocks(self.embed(x))))

    def __getitem__(self, idx: int) -> nn.Linear:
        # IsaacLab's exporter reads actor[0].in_features for its dummy input.
        if idx != 0:
            raise IndexError(idx)
        return self.embed


class SimbaActorCritic(ActorCritic):
    """Keep RSL RL's PPO and Gaussian policy plumbing while replacing its MLPs."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_hidden_dim: int = 128,
        actor_num_blocks: int = 2,
        critic_hidden_dim: int = 256,
        critic_num_blocks: int = 2,
        simba_depth_scale: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        actor_obs_dim = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        critic_obs_dim = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        self.actor = _SimbaTrunk(
            actor_obs_dim, num_actions, actor_hidden_dim, actor_num_blocks, 0.01, simba_depth_scale
        )
        self.critic = _SimbaTrunk(
            critic_obs_dim, 1, critic_hidden_dim, critic_num_blocks, 1.0, simba_depth_scale
        )
        print(f"SimBa Actor: {self.actor}")
        print(f"SimBa Critic: {self.critic}")
