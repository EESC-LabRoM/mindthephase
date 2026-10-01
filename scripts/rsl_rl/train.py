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
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=None,
    help="Investigator: Gram/Jacobian/phase metrics every N iterations (the last iteration is always measured).",
)
parser.add_argument(
    "--phase_min_samples",
    type=int,
    default=None,
    help="Investigator: minimum live states per contact phase for its rank estimates.",
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
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import the standard IsaacLab tasks. Register MyRelic tasks only when requested.
import isaaclab_tasks  # noqa: F401  # registers the built-in IsaacLab tasks
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
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if agent_cfg.logger == "wandb" and not agent_cfg.wandb_project:
        raise ValueError("Set WANDB_PROJECT or pass --log_project_name.")
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
        raise FileExistsError(
            f"{log_dir} already exists. Choose a new --name, or pass --resume to continue it."
        )

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

        # Upload completed videos to wandb at the training iteration being collected,
        # so they share the row RSL-RL logs with step=iteration.
        import wandb as _wandb
        _rec = env  # bind RecordVideo instance before env gets reassigned
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
        resume_path = cli_args.find_checkpoint(
            log_root_path, args_cli.task, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    video_start_iter = runner.current_learning_iteration

    # dump the configuration and task identity into the log directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)
    with open(os.path.join(log_dir, cli_args.TASK_FILE), "w") as f:
        f.write(args_cli.task + "\n")

    # -- Investigator setup (representation rank analysis) --
    is_spot_task = args_cli.task.startswith("Isaac-Spot-Locomotion-")
    investigator_cfg = InvestigatorCfg()
    if args_cli.checkpoint_interval is not None:
        investigator_cfg.checkpoint_interval = args_cli.checkpoint_interval
    if args_cli.phase_min_samples is not None:
        investigator_cfg.min_samples_per_phase = args_cli.phase_min_samples
    investigator = Investigator(runner, cfg=investigator_cfg)
    phase_fn, phase_contact_count = make_contact_phase_fn(env, args_cli.task)
    investigator.install(
        # Spot policy obs 9:12 are the velocity commands (vx, vy, yaw rate).
        sort_fn=(lambda obs: obs[:, 9:12].norm(dim=-1)) if is_spot_task else None,
        sort_label=r"$|c_v|$",
        phase_fn=phase_fn,
        phase_contact_count=phase_contact_count,
    )

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # post-hoc matplotlib report, then upload it with the analysis artifact
    investigator.generate_report()
    investigator.upload_artifact()

    # Finish logging before simulation_app.close(): Isaac Sim's fast shutdown skips
    # Python atexit, which would otherwise drop the final W&B row and exit record.
    if runner.writer is not None:
        if runner.logger_type in ("wandb", "neptune"):
            runner.writer.stop()
        else:
            runner.writer.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
