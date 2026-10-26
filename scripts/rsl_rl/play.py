# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Play an RSL-RL checkpoint."""


import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL checkpoint with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record a playback video."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--center", action="store_true", default=False, help="Look at the robot."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os

import gymnasium as gym

import mind.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm


def main():
    """Play with RSL-RL agent."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    if args_cli.center:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, 3.0, 3.0)
        env_cfg.viewer.resolution = (1920, 1080)

    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl"))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = cli_args.find_checkpoint(
            log_root_path, args_cli.task, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    log_dir = os.path.dirname(resume_path)

    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
    )

    obs = env.get_observations()
    timestep = 0
    progress_bar = (
        tqdm(total=args_cli.video_length, desc="Recording video")
        if args_cli.video
        else None
    )
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            progress_bar.update(1)
            if timestep == args_cli.video_length:
                break
    if progress_bar is not None:
        progress_bar.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
