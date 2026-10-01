# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
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
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--center", action="store_true", default=False, help="Look at the robot.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import copy
import os

import gymnasium as gym
import onnxruntime as ort

# Import extensions to set up environment tasks
import mind.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils import parse_env_cfg
from mind.assets.spot.constants import (
    ORDERED_JOINT_NAMES_SPOT_ARM,
    ORDERED_JOINT_NAMES_SPOT_BASE,
    SPOT_DEFAULT_JOINT_POS,
)
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm


def export_policy_as_onnx(
    policy: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    scale=1.0,
    shift=None,
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose, scale, shift)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False, scale=1.0, shift=None):
        super().__init__()
        self.verbose = verbose
        self.scale = float(scale)

        assert not policy.is_recurrent
        # copy policy parameters
        assert hasattr(policy, "actor")
        self.actor = copy.deepcopy(policy.actor)
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        if shift is None:
            dummy_shift = torch.zeros_like(self.actor(self.normalizer(torch.zeros(1, self.actor[0].in_features))))
            self.register_buffer("shift", dummy_shift)
        else:
            shift = torch.tensor(shift, dtype=torch.float32)
            self.register_buffer("shift", shift)

    def forward(
        self,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos,
        joint_vel,
        last_actions,
        height_commands,
        base_orientation_commands,
    ):
        # Same term order as the policy observation group (65 dims for Spot).
        obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                velocity_commands,
                joint_pos-self.shift,
                joint_vel,
                (last_actions - self.shift[:12])/self.scale,
                height_commands,
                base_orientation_commands,
            ],
            dim=-1,
        )
        return self.actor(self.normalizer(obs)) * self.scale + self.shift[:12]

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        # Define dummy inputs for each term
        dummy_inputs = (
            torch.zeros(1, 3),  # base_lin_vel
            torch.zeros(1, 3),  # base_ang_vel
            torch.zeros(1, 3),  # projected_gravity
            torch.zeros(1, 3),  # velocity_commands
            torch.zeros(1, 19), # joint_pos
            torch.zeros(1, 19), # joint_vel
            torch.zeros(1, 12), # actions
            torch.zeros(1, 1),  # height_commands
            torch.zeros(1, 2),  # base_orientation_commands
        )
        input_names = [
            "base_linear_velocity",
            "base_angular_velocity",
            "projected_gravity",
            "velocity_commands",
            "joint_positions",
            "joint_velocities",
            "last_actions",
            "height_commands",
            "base_orientation_commands",
        ]
        torch.onnx.export(
            self,
            dummy_inputs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=input_names,
            output_names=["actions_output"],
            dynamic_axes={name: {0: "batch_size"} for name in input_names},
        )


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.center:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, 3.0, 3.0)
        env_cfg.viewer.resolution = (1920, 1080)  # (4096, 2160)

    # save original scale and offset
    original_scale = env_cfg.actions.joint_pos.scale
    offset = [SPOT_DEFAULT_JOINT_POS[name] for name in ORDERED_JOINT_NAMES_SPOT_BASE+ORDERED_JOINT_NAMES_SPOT_ARM]

    env_cfg.actions.joint_pos.scale = 1.0
    env_cfg.observations.policy.joint_pos.params["is_relative"] = False
    assert original_scale == 0.2
    env_cfg.actions.joint_pos.use_default_offset = False   


    # resolve checkpoint path (flat layout: logs/rsl_rl/<run>/)
    if args_cli.checkpoint is not None and os.path.isfile(args_cli.checkpoint):
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint from direct path: {resume_path}")
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl"))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = cli_args.find_checkpoint(
            log_root_path, args_cli.task, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # After runner initialization, toggle the observation manager to return dictionary observations
    # This avoids breaking RSL-RL's internal model initialization which expects flat tensors.
    env.unwrapped.observation_manager._group_obs_concatenate["policy"] = False

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy_nn,
        normalizer=normalizer,
        path=export_model_dir,
        filename="policy2.onnx",
        scale=original_scale,
        shift=offset,
    )

    # load the exported onnx model
    onnx_model_path = os.path.join(export_model_dir, "policy2.onnx")
    ort_session = ort.InferenceSession(onnx_model_path)


    # reset environment
    obs = env.get_observations()
    timestep = 0
    progress_bar = tqdm(total=args_cli.video_length, desc="Recording video") if args_cli.video else None
    # simulate environment
    print("Original scale: ", original_scale)
    print("Offset: ", offset)
    assert env.unwrapped.action_manager.get_term("joint_pos")._scale == 1.0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # policy group as a dict of terms (absolute joint positions, see above)
            obs_dict = obs["policy"]
            # Convert dict of tensors to dict of numpy arrays for ONNX
            onnx_inputs = {
                "base_linear_velocity": obs_dict["base_lin_vel"].detach().cpu().numpy(),
                "base_angular_velocity": obs_dict["base_ang_vel"].detach().cpu().numpy(),
                "projected_gravity": obs_dict["projected_gravity"].detach().cpu().numpy(),
                "velocity_commands": obs_dict["velocity_commands"].detach().cpu().numpy(),
                "joint_positions": obs_dict["joint_pos"].detach().cpu().numpy(),
                "joint_velocities": obs_dict["joint_vel"].detach().cpu().numpy(),
                "last_actions": obs_dict["actions"].detach().cpu().numpy(),
                "height_commands": obs_dict["height_commands"].detach().cpu().numpy(),
                "base_orientation_commands": obs_dict["base_orientation_commands"].detach().cpu().numpy(),
            }
            onnx_outputs = ort_session.run(None, onnx_inputs)
            actions = torch.from_numpy(onnx_outputs[0]).to(device=env.unwrapped.device)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            progress_bar.update(1)
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    if progress_bar is not None:
        progress_bar.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
