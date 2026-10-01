# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group(
        "rsl_rl", description="Arguments for RSL-RL agent."
    )
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix to the log directory.",
    )
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to resume from a checkpoint (--resume / --no-resume).",
    )
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file to resume from."
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )


def parse_rsl_rl_cfg(
    task_name: str, args_cli: argparse.Namespace
) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
        task_name, "rsl_rl_cfg_entry_point"
    )
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


TASK_FILE = os.path.join("params", "task.txt")
"""Per-run file recording the training task id, written by train.py."""


def training_task(task_name: str) -> str:
    """Map a play task id to the training task whose checkpoints it loads."""
    return task_name.replace("-Play-v", "-v")


def find_checkpoint(log_root: str, task_name: str, load_run: str, checkpoint: str) -> str:
    """Resolve a checkpoint in the flat ``logs/rsl_rl/<run>/`` layout.

    The most recently modified run whose name matches ``load_run`` and whose
    recorded training task matches ``task_name`` is selected, so runs of other tasks
    and analysis directories are never picked by default. Within it, the newest
    checkpoint matching ``checkpoint`` is returned (``model_10`` after ``model_9``).
    """
    task = training_task(task_name)
    candidates = []
    for entry in os.scandir(log_root):
        task_file = os.path.join(entry.path, TASK_FILE)
        if not (entry.is_dir() and re.match(load_run, entry.name) and os.path.isfile(task_file)):
            continue
        with open(task_file) as f:
            if f.read().strip() == task:
                candidates.append(entry)
    if not candidates:
        raise ValueError(
            f"No run in {log_root} matches --load_run {load_run!r} with task {task!r} "
            f"(recorded in <run>/{TASK_FILE}). Pass --checkpoint <path> for older runs."
        )
    run = max(candidates, key=lambda e: e.stat().st_mtime)
    models = [f for f in os.listdir(run.path) if re.match(checkpoint, f)]
    if not models:
        raise ValueError(f"No checkpoints in {run.path} match {checkpoint!r}.")
    return os.path.join(run.path, max(models, key=lambda m: f"{m:0>15}"))


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if agent_cfg.logger == "wandb":
        # train.py rejects a missing project; playback never opens a W&B run.
        agent_cfg.wandb_project = args_cli.log_project_name or os.environ.get("WANDB_PROJECT")
        entity = os.environ.get("WANDB_ENTITY")
        if entity:
            # RSL-RL 3.0.1 reads WANDB_USERNAME rather than a config entity.
            os.environ["WANDB_USERNAME"] = entity

    return agent_cfg
