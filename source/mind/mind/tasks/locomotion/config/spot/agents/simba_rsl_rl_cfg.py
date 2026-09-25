"""SimBa PPO runner for the local Spot locomotion task."""

from isaaclab.utils import configclass

from mind.rsl_rl.simba_cfg import SimbaPolicyCfg

from .rsl_rl_cfg import SpotLocomotionPPORunnerCfg


@configclass
class SpotLocomotionSimbaPPORunnerCfg(SpotLocomotionPPORunnerCfg):
    """Keep Spot's PPO settings and use the shared SimBa actor and critic."""

    experiment_name = "spot_locomotion_simba"
    logger = "tensorboard"
    policy: SimbaPolicyCfg = SimbaPolicyCfg()
