## Install

To install this project, you can use the `uv` package manager.

1. **Install UV**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. **Install dependencies**

    ```bash
    uv sync
    ```

## Train and Play

```bash
uv run scripts/rsl_rl/train.py --task Isaac-Spot-Locomotion-v0 --headless
uv run scripts/rsl_rl/play.py --task Isaac-Spot-Locomotion-Play-v0 --center
```

TODO: include the weights used for deploying
TODO: include the code to setup investigator

## Repository Organization

The main Python package is located at `source/relic/relic/`. The project functionalities are organized into the following key directories:

- `tasks/`: Contains implementations for the Reinforcement Learning tasks (e.g., `locomotion`). Each task directory typically contains:
  - `config/`: Environment configurations tailored for specific robots (e.g., `spot`).
  - `mdp/`: Holds all default templates and functions defining the task's Markov Decision Process (MDP), including rewards, observations, and commands.
- `actuators/`: Contains implementations and configurations for robot-specific actuators.
- `assets/`: Stores asset definitions, robot models (URDF/USD), configuration constants, and pre-trained weights.

## Third-Party Code

We use RSL_RL for RL training and adapt the following scripts from [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate/tree/main?tab=readme-ov-file#installation).

Spot simulation was based on the [Relic](https://github.com/rai-opensource/relic) setup.

