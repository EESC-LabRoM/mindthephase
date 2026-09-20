import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def left_right_symmetry_base_lin_vel(obs: torch.Tensor):
    """Applies left-right symmetry to the base linear velocity observation.

    For left-right symmetry, the y-component of the base linear velocity is negated.
    """
    obs_sym = obs.clone()
    obs_sym[..., 1] = -obs_sym[..., 1]  # vy
    return obs_sym


def left_right_symmetry_base_ang_vel(obs: torch.Tensor):
    """Applies left-right symmetry to the base angular velocity observation.

    For left-right symmetry, the y and z components of the base angular velocity are negated.
    """
    obs_sym = obs.clone()
    obs_sym[..., 0] = -obs_sym[..., 0]  # wx
    obs_sym[..., 2] = -obs_sym[..., 2]  # wz
    return obs_sym


def left_right_symmetry_projected_gravity(obs: torch.Tensor):
    """Applies left-right symmetry to the projected gravity observation.

    For left-right symmetry, the y-component of the projected gravity is negated.
    """
    obs_sym = obs.clone()
    obs_sym[..., 1] = -obs_sym[..., 1]  # gy
    return obs_sym


def left_right_symmetry_velocity_commands(obs: torch.Tensor):
    """Applies left-right symmetry to the velocity commands observation.

    For left-right symmetry, the y-component of the linear velocity and the z-component of the angular velocity
    are negated.
    """
    obs_sym = obs.clone()
    obs_sym[..., 1] = -obs_sym[..., 1]  # vy
    obs_sym[..., 2] = -obs_sym[..., 2]  # wz
    return obs_sym


def left_right_symmetry_base_joint_pos(obs: torch.Tensor):
    """Applies left-right symmetry to the joint position observation.

    # Swap left and right leg joints
    # LF -> RF, LH -> RH
    # 0, 1, 2 (LF) <-> 3, 4, 5 (RF)
    # 6, 7, 8 (LH) <-> 9, 10, 11 (RH)

    # Negate hip joint for symmetry

    Be sure the joint order in observation and action are this one
    """
    obs_sym = obs.clone()

    swapped_obs = torch.cat(
        (
            obs_sym[..., 3:6],
            obs_sym[..., 0:3],
            obs_sym[..., 9:12],
            obs_sym[..., 6:9],
        ),
        dim=-1,
    )

    swapped_obs[..., ::3] = -swapped_obs[..., ::3]  # Negate hip joints

    return swapped_obs


def left_right_symmetry_arm_joints(obs: torch.Tensor):
    """Applies left-right symmetry to the arm joint position observation.

    Indices to negate inside the arm vector:
    0: arm_sh0 (Z-axis/Yaw)
    3: arm_el1 (X-axis/Roll)
    5: arm_wr1 (X-axis/Roll)
    """
    swapped_arm = obs.clone()
    arm_negation_indices = [0, 3, 5]
    swapped_arm[..., arm_negation_indices] = -swapped_arm[..., arm_negation_indices]
    return swapped_arm


def left_right_symmetry_arm_joint_vel(obs: torch.Tensor):
    """Applies left-right symmetry to the arm joint velocity observation."""
    return left_right_symmetry_arm_joints(obs)


def left_right_symmetry_base_joint_vel(obs: torch.Tensor):
    """Applies left-right symmetry to the joint velocity observation."""
    return left_right_symmetry_base_joint_pos(obs)


def left_right_symmetry_actions(obs: torch.Tensor):
    """Applies left-right symmetry to the actions observation."""
    return left_right_symmetry_base_joint_pos(obs)


def left_right_symmetry_height_scan(
    obs: torch.Tensor, dim_x: int, dim_y: int, ordering: str = "xy"
):
    """Applies left-right symmetry to the height scan observation.

    For left-right symmetry, the y-coordinates of the height scan points are flipped.
    """
    # Get the number of environments
    num_envs = obs.shape[0]
    obs_sym = obs.clone()

    assert ordering == "xy", (
        "Only 'xy' ordering is supported in left-right symmetry for height scan."
    )

    # 1. Reshape from (num_envs, num_rays * 3) to (num_envs, num_rays)
    obs_unflattened = obs_sym.reshape(num_envs, dim_y, dim_x)
    # 2. Flip the order of the rays along the num_rays dimension (dim=1)
    obs_unflattened = obs_unflattened.flip(dims=[1])
    # 4. Flatten back to the original shape (num_envs, num_rays * 3)
    obs_sym_flat = obs_unflattened.reshape(num_envs, -1)

    return obs_sym_flat


def height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    # robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # robot: Articulation = env.scene[robot_cfg.name]

    z_data = (
        sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    )
    is_nan = torch.isinf(z_data)

    z_data[is_nan] = -1.0
    return z_data


def height_scan_xyz(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.0
) -> torch.Tensor:
    """Returns XYZ readings from the height scanner in shape (num_envs, num_rays*3),
    relative to the sensor origin, applying the offset on the Z coordinate.

    num_envs (N): Number of environments
    num_rays (B): Number of rays in the height scanner
    """

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # Sensor position in world frame -> shape: (N, 3)
    sensor_pos_z = sensor.data.pos_w[:, 2]
    z_data = sensor_pos_z.unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    is_nan = torch.isinf(z_data)
    z_data[is_nan] = -1.0

    # Get the local coordinates of the rays
    # Ray hit points in world frame -> shape: (N, B, 3)
    relative_xyz = sensor.data.ray_hits_w.clone()
    relative_xyz[..., 0] = sensor.ray_starts[..., 0]
    relative_xyz[..., 1] = sensor.ray_starts[..., 1]
    relative_xyz[..., 2] = z_data

    # Flatten to shape: (N, num_rays*3)
    hits_flat = relative_xyz.reshape(relative_xyz.shape[0], -1)

    return hits_flat


def left_right_symmetry_height_scan_ame(
    obs: torch.Tensor, dim_x: int, dim_y: int, ordering: str = "xy"
):
    """Applies left-right symmetry to the height scan observation.

    For left-right symmetry, the y-coordinates of the height scan points are flipped.
    """
    # Get the number of environments
    num_envs = obs.shape[0]
    obs_sym = obs.clone()

    assert ordering == "xy", (
        "Only 'xy' ordering is supported in left-right symmetry for height scan."
    )

    obs_sym[..., 2] = left_right_symmetry_height_scan(
        obs=obs[..., 2].clone(), dim_x=dim_y, dim_y=dim_x, ordering=ordering
    )

    return obs_sym


def left_right_symmetry_height_commands(obs: torch.Tensor) -> torch.Tensor:
    """Applies left-right symmetry to the height commands observation.

    Height is invariant under left-right reflection.
    """
    return obs.clone()


def left_right_symmetry_base_orientation_commands(obs: torch.Tensor) -> torch.Tensor:
    """Applies left-right symmetry to the base orientation commands observation.

    The orientation command consists of [roll, pitch].
    Under left-right reflection (across X-Z plane), roll is negated and pitch is invariant.
    """
    obs_sym = obs.clone()
    obs_sym[..., 0] = -obs_sym[..., 0]  # negate roll
    return obs_sym


def left_right_symmetry_robot_base_height(obs: torch.Tensor) -> torch.Tensor:
    """Applies left-right symmetry to the robot base height observation.

    Height is invariant under left-right reflection.
    """
    return obs.clone()

