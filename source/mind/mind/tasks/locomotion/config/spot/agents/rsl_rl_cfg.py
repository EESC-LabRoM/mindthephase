from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from mind.tasks.locomotion.locomotion_env_cfg import (
    left_right_symmetry_obs,
)

import os
from datetime import datetime
from dataclasses import field
from dotenv import load_dotenv

load_dotenv()


@configclass
class SpotLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the PPO runner."""

    num_steps_per_env = 24
    max_iterations = 10_000
    save_interval = 200
    experiment_name = "mind" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name: str = field(
        default_factory=lambda: f"mind_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logger = "wandb"
    wandb_project = os.environ.get("WANDB_PROJECT") or "mindthephase"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
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
