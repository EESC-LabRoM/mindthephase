# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
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
    help="Run name -- becomes the log folder under logs/rsl_rl/ and the wandb run name. Falls back to a timestamp.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import mind.tasks  # noqa: F401
from investigator import Investigator, InvestigatorCfg

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
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # flat run layout: logs/rsl_rl/{name}/
    # name is the primary key for a run; falls back to a timestamp when --name is omitted.
    run_name = args_cli.name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg.experiment_name = "rsl_rl"
    agent_cfg.run_name = run_name

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl"))
    log_dir = os.path.join(log_root_path, run_name)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    if os.path.exists(log_dir) and not agent_cfg.resume:
        print(f"[WARN] {log_dir} already exists -- existing files may be overwritten.")

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
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

        # Upload completed videos to wandb
        import wandb as _wandb
        _rec = env  # bind RecordVideo instance before env gets reassigned
        _orig_stop = _rec.stop_recording
        def _stop_and_log():
            name = _rec._video_name
            _orig_stop()
            if name is not None:
                path = os.path.join(_rec.video_folder, f"{name}.mp4")
                if os.path.exists(path) and _wandb.run is not None:
                    _wandb.log({"train/video": _wandb.Video(path, fps=30, format="mp4")})
        _rec.stop_recording = _stop_and_log

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory in a background thread
    import threading
    def _save_configs():
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
            pickle.dump(env_cfg, f)
        with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
            pickle.dump(agent_cfg, f)

    threading.Thread(target=_save_configs, daemon=True).start()

    # -- Investigator setup (representation rank analysis) --
    investigator = Investigator(runner, cfg=InvestigatorCfg())
    investigator.install(sort_fn=lambda obs: obs[:, 9:12].norm(dim=-1))

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # post-hoc matplotlib report (supplementary to wandb)
    investigator.generate_report()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
