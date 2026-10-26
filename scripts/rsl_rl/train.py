# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""


import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=400,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=82000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help=(
        "Run name -- log folder under logs/rsl_rl/ and W&B run name. "
        "Falls back to a timestamp."
    ),
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=None,
    help=(
        "Investigator: Gram/Jacobian/phase metrics every N iterations "
        "(the last iteration is always measured)."
    ),
)
parser.add_argument(
    "--phase_min_samples",
    type=int,
    default=None,
    help="Investigator: minimum live states per contact phase for its rank estimates.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import pickle
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
if args_cli.task and (
    args_cli.task.startswith(("Isaac-Spot-Locomotion-", "Isaac-Velocity-Flat-Spot-MLP-"))
    or args_cli.task.endswith("-Simba-v0")
):
    import mind.tasks  # noqa: F401
from investigator import Investigator, InvestigatorCfg, make_contact_phase_fn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    run_name = args_cli.name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg.experiment_name = "rsl_rl"
    agent_cfg.run_name = run_name

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl"))
    log_dir = os.path.join(log_root_path, run_name)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    if os.path.exists(log_dir) and not agent_cfg.resume:
        raise FileExistsError(
            f"{log_dir} already exists. Choose a new --name, or pass --resume to continue it."
        )

    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

        import wandb as _wandb
        _rec = env
        _orig_stop = _rec.stop_recording
        def _stop_and_log():
            name = _rec._video_name
            _orig_stop()
            if name is not None:
                path = os.path.join(_rec.video_folder, f"{name}.mp4")
                if os.path.exists(path) and _wandb.run is not None:
                    iteration = video_start_iter + (_rec.step_id - 1) // agent_cfg.num_steps_per_env
                    _wandb.log(
                        {"train/video": _wandb.Video(path, fps=30, format="mp4")},
                        step=iteration,
                    )
        _rec.stop_recording = _stop_and_log

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        resume_path = cli_args.find_checkpoint(
            log_root_path, args_cli.task, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
    video_start_iter = runner.current_learning_iteration

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)
    with open(os.path.join(log_dir, cli_args.TASK_FILE), "w") as f:
        f.write(args_cli.task + "\n")

    is_spot_task = args_cli.task.startswith("Isaac-Spot-Locomotion-")
    investigator_cfg = InvestigatorCfg()
    if args_cli.checkpoint_interval is not None:
        investigator_cfg.checkpoint_interval = args_cli.checkpoint_interval
    if args_cli.phase_min_samples is not None:
        investigator_cfg.min_samples_per_phase = args_cli.phase_min_samples
    investigator = Investigator(runner, cfg=investigator_cfg)
    phase_fn, phase_contact_count = make_contact_phase_fn(env, args_cli.task)
    investigator.install(
        sort_fn=(lambda obs: obs[:, 9:12].norm(dim=-1)) if is_spot_task else None,
        sort_label=r"$|c_v|$",
        phase_fn=phase_fn,
        phase_contact_count=phase_contact_count,
    )

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    investigator.generate_report()
    investigator.upload_artifact()

    if runner.writer is not None:
        if runner.logger_type in ("wandb", "neptune"):
            runner.writer.stop()
        else:
            runner.writer.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
