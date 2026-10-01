## Install

To install this project, you can use the `uv` package manager.

1. **Install UV**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. **Install dependencies**

    ```bash
    git submodule update --init IsaacLab
    uv sync --locked
    ```

## W&B setup

```bash
uv run wandb login
export WANDB_PROJECT="<your-project>"
# Optional team; otherwise use the authenticated account.
export WANDB_ENTITY="<your-team>"
```

Use `--logger wandb`; project comes from `WANDB_PROJECT` or `--log_project_name`.
Keep credentials out of the repository.

Accept the NVIDIA Omniverse EULA on first use; afterward set
`OMNI_KIT_ACCEPT_EULA=YES` for noninteractive runs.

## Train and Play

```bash
uv run scripts/rsl_rl/train.py --task Isaac-Spot-Locomotion-v0 --headless
uv run scripts/rsl_rl/play.py --task Isaac-Spot-Locomotion-Play-v0 --center
```

TODO: include the weights used for deploying

## ANYmal D, G1, and Spot with Investigator

The local training script attaches [Investigator](scripts/rsl_rl/investigator.py) automatically.

```bash
uv run scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Anymal-D-v0 --headless

uv run scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 --headless
```

For the Simba variations:

```bash
uv run scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Anymal-D-Simba-v0 --headless

uv run scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-Simba-v0 --headless

uv run scripts/rsl_rl/train.py \
  --task Isaac-Spot-Locomotion-Simba-v0 --headless
```

When the task's contact sensor exposes the expected feet, Investigator also logs contact-conditioned metrics every 500 iterations. It groups current observations by the number of feet with upward contact force above 1 N, then measures feature, Gram, and Jacobian ranks per group in `phase_rank_timeseries.csv` and `history.json`.

## Repository Organization

The main Python package is located at `source/mind/mind/`. The project functionalities are organized into the following key directories:

- `tasks/`: Contains implementations for the Reinforcement Learning tasks (e.g., `locomotion`). Each task directory typically contains:
  - `config/`: Environment configurations tailored for specific robots (e.g., `spot`).
  - `mdp/`: Holds all default templates and functions defining the task's Markov Decision Process (MDP), including rewards, observations, and commands.
- `actuators/`: Contains implementations and configurations for robot-specific actuators.
- `assets/`: Stores asset definitions, robot models (URDF/USD), configuration constants, and pre-trained weights.

## Third-Party Code

We use RSL_RL for RL training and adapt the following scripts from [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate/tree/main?tab=readme-ov-file#installation).

Spot simulation was based on the [Relic](https://github.com/rai-opensource/relic) setup.
