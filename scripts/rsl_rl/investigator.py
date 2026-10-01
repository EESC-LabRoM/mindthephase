"""
investigator.py — Self-contained observability engine for legged locomotion rank analysis.

Integration (in train.py):

    from investigator import Investigator, InvestigatorCfg

    investigator = Investigator(runner, cfg=InvestigatorCfg())
    investigator.install(sort_fn=lambda obs: obs[:, 9:12].norm(dim=-1))
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    investigator.generate_report()

Designed for rsl_rl with IsaacLab. Actor/Critic accessed via runner.alg.policy.actor/.critic.
"""

from __future__ import annotations

import json
import csv
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from functools import wraps
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from paper_style import (
    CMAP_DIVERGING, CMAP_SEQUENTIAL, COLUMN_WIDTH, FULL_WIDTH, PALETTE,
    arch_of, figure_image, paper_name, run_styles, save_figure, style_axes,
)

# [WANDB INTEGRATION] soft-import; all wandb calls below are gated on _WANDB_AVAILABLE
_WANDB_AVAILABLE = False
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# Policy input/output labels for plots
# ===================================================================

# Short symbols for observation terms (paper notation); unknown terms use initials.
OBS_SYMBOLS = {
    "base_lin_vel": r"$v$",
    "base_ang_vel": r"$\omega$",
    "projected_gravity": r"$g$",
    "velocity_commands": r"$c_v$",
    "joint_pos": r"$q$",
    "joint_vel": r"$\dot{q}$",
    "actions": r"$a_{t-1}$",
    "last_action": r"$a_{t-1}$",
    "height_commands": r"$c_h$",
    "base_orientation_commands": r"$c_\theta$",
    "height_scan": r"$h$",
}

_JOINT_TOKENS = {"left": "L", "right": "R", "pitch": "P", "roll": "R", "yaw": "Y"}


def obs_symbol(term: str) -> str:
    return OBS_SYMBOLS.get(term, "".join(t[0] for t in term.split("_") if t))


def short_joint_name(name: str) -> str:
    """`fl_hx` -> `FL hx`, `LF_HAA` -> `LF HAA`, `left_hip_pitch_joint` -> `L hip P`."""
    tokens = [t for t in name.removesuffix("_joint").split("_") if t]
    out = []
    for i, t in enumerate(tokens):
        if t.lower() in _JOINT_TOKENS:
            out.append(_JOINT_TOKENS[t.lower()])
        elif i == 0 and len(t) == 2 and len(tokens) > 1:
            out.append(t.upper())  # leg prefix (fl, rh, ...)
        else:
            out.append(t)
    return " ".join(out)


def policy_layout(env, policy, obs_dim: int, action_dim: int) -> dict:
    """Action joint labels and observation term spans of the actor's input/output.

    Falls back to index labels when the managers do not match the actor's dims.
    """
    layout = {"actions": [f"$a_{{{i}}}$" for i in range(action_dim)], "obs_groups": []}
    base = getattr(env, "unwrapped", env)
    om = getattr(base, "observation_manager", None)
    am = getattr(base, "action_manager", None)
    if om is not None:
        groups = getattr(policy, "obs_groups", {}).get("policy", ["policy"])
        spans, start = [], 0
        for g in groups:
            for term, shape in zip(om.active_terms[g], om.group_obs_term_dim[g]):
                dim = int(np.prod(shape))
                spans.append((obs_symbol(term), start, start + dim))
                start += dim
        if start == obs_dim:
            layout["obs_groups"] = spans
    if am is not None:
        names = []
        for t in am.active_terms:
            term = am.get_term(t)
            if term.action_dim:
                names += list(getattr(term, "_joint_names", []))[:term.action_dim]
        if len(names) == action_dim:
            layout["actions"] = [short_joint_name(n) for n in names]
    return layout


# Contact-count groups used for phase-conditioned rank measurements.
PHASE_GROUPS = {
    4: {
        "pooled": None,
        "stance": (3, 4),
        "swing": (0, 1),
        "double_support": (2,),
    },
    2: {
        "pooled": None,
        "flight": (0,),
        "single_support": (1,),
        "double_support": (2,),
    },
}

CONTACT_COUNT_LABELS = {
    4: ("flight", "swing_dominant", "double_support", "stance_dominant", "full_stance"),
    2: ("flight", "single_support", "double_support"),
}

PHASE_CSV_FIELDS = (
    "global_step", "phase", "sample_count", "valid", "feature_erank",
    "pca_rank_99", "gram_erank", "jacobian_erank",
    "jacobian_local_erank_mean", "jacobian_local_erank_std", "jacobian_dof_gap",
)


def make_contact_phase_fn(env, task_name: str | None):
    """Find a task's foot contacts and return a support-count callback."""
    if not task_name:
        return None, None
    if "Anymal-D" in task_name:
        pattern, expected = ".*FOOT", 4
    elif "G1" in task_name:
        pattern, expected = ".*_ankle_roll_link", 2
    elif "Spot" in task_name:
        pattern, expected = ".*_foot", 4
    else:
        return None, None

    sensor = env.unwrapped.scene.sensors.get("contact_forces")
    if sensor is None:
        warnings.warn(f"[Investigator] {task_name}: contact_forces sensor unavailable; phase metrics disabled.")
        return None, None
    try:
        foot_ids, foot_names = sensor.find_bodies(pattern)
    except ValueError as error:
        warnings.warn(f"[Investigator] {task_name}: {error}; phase metrics disabled.")
        return None, None
    if len(foot_ids) != expected:
        warnings.warn(
            f"[Investigator] {task_name}: expected {expected} feet matching {pattern!r}, "
            f"found {foot_names}; phase metrics disabled."
        )
        return None, None

    ids = torch.tensor(foot_ids, dtype=torch.long, device=env.unwrapped.device)

    def phase_fn(current_env):
        forces = current_env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w
        return forces[:, ids, 2].gt(1.0).sum(dim=-1)

    print(f"[Investigator] Phase contacts: {', '.join(foot_names)}")
    return phase_fn, expected


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class InvestigatorCfg:
    """All knobs for the Investigator."""

    enabled: bool = True

    # -- Cadence --
    log_interval: int = 100
    """Compute live representation metrics every N training iterations."""
    checkpoint_interval: int = 500
    """Compute expensive metrics (Gram, Jacobian, native plots) every N iterations."""
    weight_rank_interval: int = 250
    """Compute per-layer weight erank every N iterations."""

    # -- Eval observations --
    n_eval_obs: int = 2048
    """Fixed eval observation count for Gram / Jacobian analysis."""

    # -- Jacobian --
    jacobian_batch_size: int = 64
    """Subset of eval obs for Jacobian erank (O(action_dim) backward passes per sample)."""
    jacobian_use_vmap: bool = True
    """Use torch.func.jacrev + vmap (PyTorch >= 2.0)."""

    # -- Saving (local disk backup) --
    save_features: bool = True
    save_grams: bool = True
    save_spectra: bool = True
    save_jacobian_spectra: bool = True
    save_jacobians: bool = True
    """Save full J_mean matrix (not just SVs) for post-hoc heatmaps."""

    # -- Checkpoint heatmaps: saved to investigator/plots/checkpoints/, and logged to W&B when a run exists --
    gram_heatmap: bool = True
    feature_heatmap: bool = True
    """Feature activation heatmap (samples x head-input units)."""
    jacobian_heatmap: bool = True
    """Mean and std Jacobian heatmaps (actions x observations)."""
    weight_heatmap: bool = True
    """Per-layer weight heatmaps."""

    # -- WandB --
    wandb_log: bool = True
    wandb_spectral_histograms: bool = True
    """Log wandb.Histogram for SV distributions (produces heatmap-over-time)."""
    wandb_artifact_on_end: bool = True
    """Upload investigator/ dir as wandb Artifact at end of training."""
    wandb_summary_on_end: bool = True
    """Write final summary metrics to wandb.run.summary."""

    # -- Thresholds --
    dead_neuron_threshold: float = 1e-1
    erank_eps: float = 1e-10

    # -- Gait phase metrics --
    phase_log_enabled: bool = True
    """Measure representation ranks by foot-contact count when a contact sensor is available."""
    min_samples_per_phase: int = 128
    """Minimum observations needed for a phase's rank estimates."""

    # -- Scrape from rsl_rl's learn() locals for post-hoc comparison --
    # Scalars (float/int) go as-is. Deques/lists are reduced with mean.
    # Dicts are flattened one level (e.g. loss_dict -> investigator/train/value_loss).
    locs_scalar_keys: tuple[str, ...] = (
        "rewbuffer",         # deque of episode rewards -> mean_reward
        "lenbuffer",         # deque of episode lengths -> mean_episode_length
        "loss_dict",         # dict: surrogate_loss, value_loss, entropy, ...
        "collection_time",
        "learn_time",
    )


# ===================================================================
# Math utilities
# ===================================================================

def effective_rank(S: torch.Tensor, eps: float = 1e-10) -> float:
    """erank(A) = exp(-sum s_bar_i log s_bar_i) where s_bar_i = sigma_i / sum sigma_j."""
    S = S[S > eps]
    if len(S) == 0:
        return 0.0
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + eps)).sum()
    return torch.exp(entropy).item()


def pca_rank(S: torch.Tensor, threshold: float = 0.99) -> int:
    """Smallest k: top-k singular values explain >= threshold variance."""
    var = S ** 2
    total = var.sum()
    if total <= 0:
        return 0
    explained = torch.cumsum(var, dim=0) / total
    return min(torch.searchsorted(explained, threshold).item() + 1, len(S))


def compute_gram_cosine(features: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity Gram matrix [N, D] -> [N, N]."""
    F_norm = F.normalize(features, dim=-1)
    return F_norm @ F_norm.T


def _to_scalar(v) -> float | None:
    """Best-effort cast to a Python float.

    - numbers -> float
    - deque/list/tuple of numbers -> mean (None if empty)
    - 0-d / 1-d torch tensors -> mean item
    - anything else -> None
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, torch.Tensor):
        if v.numel() == 0:
            return None
        return float(v.float().mean().item())
    # deque, list, tuple
    try:
        seq = list(v)
    except TypeError:
        return None
    if not seq:
        return None
    try:
        return float(np.mean([float(x) for x in seq]))
    except (TypeError, ValueError):
        return None


# ===================================================================
# Layer discovery
# ===================================================================

def _find_linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def _find_head_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    """The policy head: the last Linear layer, whose input is the representation."""
    linears = _find_linear_layers(model)
    if len(linears) < 2:
        raise ValueError(f"Need >= 2 Linear layers, found {len(linears)}.")
    return linears[-1]


# ===================================================================
# The Investigator
# ===================================================================

class Investigator:
    """Self-contained observability engine for representation rank analysis.

    Hooks into rsl_rl's OnPolicyRunner via monkey-patching.
    All metrics go to wandb; local disk is a backup.
    """

    def __init__(self, runner, cfg: InvestigatorCfg | None = None):
        self.runner = runner
        self.cfg = cfg or InvestigatorCfg()
        self.device = runner.device

        self._fixed_eval_obs: torch.Tensor | None = None
        self._sort_indices: torch.Tensor | None = None
        self._layout: dict = {"arch": "MLP", "actions": [], "obs_groups": [], "sort_label": None}
        self._phase_fn: callable | None = None
        self._phase_contact_count: int | None = None
        self._log_dir: Path | None = None
        self._history: dict[str, list] = {}
        self._installed = False
        self._final_iteration: int | None = None

    # ---------------------------------------------------------
    # Installation
    # ---------------------------------------------------------

    def install(self, sort_fn: callable | None = None,
                phase_fn: callable | None = None,
                phase_contact_count: int | None = None,
                sort_label: str | None = None):
        """Monkey-patch runner.learn() to inject analysis hooks.

        Args:
            sort_fn: Optional callable(obs_batch) -> 1D sort keys.
                     Sorts Gram matrix rows for visible block structure.
                     Example: lambda obs: obs[:, 9:12].norm(dim=-1)
            sort_label: Axis-label text for the sort key, e.g. r"$|c_v|$".
            phase_fn: Optional callable(env) -> Tensor[int] of shape [num_envs].
                      Returns the number of feet in contact per environment.
            phase_contact_count: Number of tracked feet (2 or 4), required with phase_fn.
        """
        if not self.cfg.enabled or self._installed:
            return

        if phase_fn is not None and phase_contact_count not in PHASE_GROUPS:
            raise ValueError("Phase metrics require 2 or 4 tracked contact points.")
        self._phase_fn = phase_fn
        self._phase_contact_count = phase_contact_count

        runner = self.runner

        # Output directory
        log_dir = getattr(runner, "log_dir", None) or "logs/investigator_default"
        self._log_dir = Path(log_dir) / "investigator"
        for sub in ["features", "grams", "jacobians", "spectra", "plots/checkpoints"]:
            (self._log_dir / sub).mkdir(parents=True, exist_ok=True)

        # Resuming into the same directory: keep the earlier history (up to the
        # resumed iteration, which learn() repeats) and the same fixed eval panel.
        resume_iter = runner.current_learning_iteration
        history_path = self._log_dir / "history.json"
        if history_path.exists():
            saved = json.loads(history_path.read_text())
            self._history = {
                k: [(it, v) for it, v in vals if it < resume_iter] for k, vals in saved.items()
            }
        obs_path = self._log_dir / "fixed_eval_obs.pt"
        if obs_path.exists():
            self._fixed_eval_obs = torch.load(obs_path)
            print(f"[Investigator] Reusing fixed eval obs: {self._fixed_eval_obs.shape}")
        else:
            self._collect_fixed_eval_obs()

        # Sort indices
        if sort_fn is not None:
            with torch.no_grad():
                keys = sort_fn(self._fixed_eval_obs.to(self.device))
                self._sort_indices = torch.argsort(keys).cpu()
            torch.save(self._sort_indices, self._log_dir / "sort_indices.pt")

        # Plot labels (also read by the post-hoc CLI report, which has no env)
        policy = runner.alg.policy
        _, head = _find_head_linear(self._get_actor())
        self._layout = {
            "arch": "SimBa" if "simba" in type(policy).__name__.lower() else "MLP",
            **policy_layout(runner.env, policy, self._fixed_eval_obs.shape[1], head.out_features),
            "sort_label": sort_label if sort_fn is not None else None,
        }
        (self._log_dir / "layout.json").write_text(json.dumps(self._layout, indent=2))

        # Save config locally
        with open(self._log_dir / "investigator_cfg.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2, default=str)

        # W&B config upload waits until RSL-RL creates the run inside learn().
        self._wandb_config_pushed = False

        # -- Monkey-patch learn() --
        original_learn = runner.learn
        investigator = self

        @wraps(original_learn)
        def patched_learn(num_learning_iterations: int,
                          init_at_random_ep_len: bool = False):
            # Last iteration of this learn() call: always measured, whatever the cadence.
            investigator._final_iteration = (
                runner.current_learning_iteration + num_learning_iterations - 1
            )
            original_log = runner.log

            @wraps(original_log)
            def patched_log(locs: dict, width: int = 80, pad: int = 35):
                it = locs.get("it", 0)
                investigator._on_iteration(it, locs)
                return original_log(locs, width, pad)

            runner.log = patched_log
            try:
                original_learn(num_learning_iterations, init_at_random_ep_len)
            finally:
                runner.log = original_log
                investigator._on_training_end()

        runner.learn = patched_learn
        self._installed = True
        print(f"[Investigator] Installed -> {self._log_dir}")

    # ---------------------------------------------------------
    # Fixed eval obs
    # ---------------------------------------------------------

    def _collect_fixed_eval_obs(self):
        env = self.runner.env
        n = self.cfg.n_eval_obs
        obs = self.runner.alg.policy.get_actor_obs(env.get_observations()).to(self.device)

        if obs.shape[0] >= n:
            idx = torch.randperm(obs.shape[0])[:n]
            self._fixed_eval_obs = obs[idx].cpu()
        else:
            # Clamp to available envs to avoid duplicate-state bias
            warnings.warn(
                f"[Investigator] num_envs ({obs.shape[0]}) < n_eval_obs ({n}). "
                f"Using {obs.shape[0]} eval obs instead of {n}."
            )
            self._fixed_eval_obs = obs.cpu()

        torch.save(self._fixed_eval_obs, self._log_dir / "fixed_eval_obs.pt")
        print(f"[Investigator] Fixed eval obs: {self._fixed_eval_obs.shape}")

    # ---------------------------------------------------------
    # Iteration callback
    # ---------------------------------------------------------

    def _on_iteration(self, iteration: int, locs: dict | None = None):
        cfg = self.cfg
        self._push_wandb_config()
        if locs is not None:
            self._log_locs_scalars(iteration, locs)
        is_final = iteration == self._final_iteration
        if iteration % cfg.log_interval == 0 or is_final:
            self._compute_live_metrics(iteration)
        if iteration % cfg.weight_rank_interval == 0 or is_final:
            self._compute_weight_ranks(iteration)
        if iteration % cfg.checkpoint_interval == 0 or is_final:
            self._compute_checkpoint_metrics(iteration)
            self._dump_history()

    # ---------------------------------------------------------
    # Extra scalars scraped from rsl_rl's learn() locals
    # ---------------------------------------------------------

    def _log_locs_scalars(self, iteration: int, locs: dict):
        """Extract scalar metrics from rsl_rl's learn() locals -> history.json.

        - float/int: logged as-is.
        - deque/list/tuple of numbers: reduced with mean (skip if empty).
        - dict: flattened one level (e.g. loss_dict['value_loss'] ->
          investigator/train/value_loss).
        """
        prefix = "investigator/train/"
        # Map a couple of rsl_rl's buffer names to friendlier keys.
        rename = {"rewbuffer": "mean_reward", "lenbuffer": "mean_episode_length"}
        metrics = {}

        for key in self.cfg.locs_scalar_keys:
            if key not in locs:
                continue
            val = locs[key]
            out_key = rename.get(key, key)

            if isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    scalar = _to_scalar(sub_v)
                    if scalar is not None:
                        metrics[f"{prefix}{sub_k}"] = scalar
            else:
                scalar = _to_scalar(val)
                if scalar is not None:
                    metrics[f"{prefix}{out_key}"] = scalar

        if metrics:
            self._log_metrics(metrics, iteration)

    def _push_wandb_config(self):
        """Record investigator settings and the run path once the W&B run exists."""
        if self._wandb_config_pushed or not (self.cfg.wandb_log and _WANDB_AVAILABLE and wandb.run):
            return
        wandb.config.update({"investigator": asdict(self.cfg)}, allow_val_change=True)
        # comparison.py binds local runs to these immutable "entity/project/id" paths.
        # Resuming into the same directory starts a new W&B run, so keep every segment.
        run_file = self._log_dir / "wandb_run.json"
        paths = json.loads(run_file.read_text())["paths"] if run_file.exists() else []
        if wandb.run.path not in paths:
            paths.append(wandb.run.path)
        run_file.write_text(json.dumps({"paths": paths}))
        self._wandb_config_pushed = True

    def _on_training_end(self):
        # Save history locally (json + long-format csv for pandas)
        history_path = self._dump_history()

        # [WANDB INTEGRATION] final summary scalars to run.summary
        if self.cfg.wandb_summary_on_end and _WANDB_AVAILABLE and wandb.run:
            self._write_wandb_summary()

        print(f"[Investigator] Done. History -> {history_path}")

    def upload_artifact(self):
        """Upload the investigator directory; call after generate_report()."""
        if not (self.cfg.wandb_artifact_on_end and _WANDB_AVAILABLE and wandb.run):
            return
        try:
            art = wandb.Artifact(f"investigator-{wandb.run.id}", type="analysis")
            art.add_dir(str(self._log_dir))
            wandb.log_artifact(art)
        except Exception as e:
            warnings.warn(f"[Investigator] Artifact upload failed: {e}")

    def _dump_history(self) -> Path:
        """Write scalar history to disk without waiting for training to end."""
        history_path = self._log_dir / "history.json"
        if not self._history:
            return history_path

        serializable = {k: [(it, float(v)) for it, v in vals]
                        for k, vals in self._history.items()}

        tmp_history_path = history_path.with_suffix(".json.tmp")
        with open(tmp_history_path, "w") as f:
            json.dump(serializable, f)
        tmp_history_path.replace(history_path)

        csv_path = self._log_dir / "history.csv"
        try:
            tmp_csv_path = csv_path.with_suffix(".csv.tmp")
            with open(tmp_csv_path, "w") as f:
                f.write("metric,iteration,value\n")
                for k, vals in serializable.items():
                    for it, v in vals:
                        f.write(f"{k},{it},{v}\n")
            tmp_csv_path.replace(csv_path)
        except Exception as e:
            warnings.warn(f"[Investigator] history.csv dump failed: {e}")

        return history_path

    # [WANDB INTEGRATION] writes final/min/max/mean per metric to wandb.run.summary
    def _write_wandb_summary(self):
        summary = {}
        for key in [
            "investigator/actor/feature_erank",
            "investigator/actor/feature_erank_fixed",
            "investigator/gram/erank",
            "investigator/policy/jacobian_erank",
            "investigator/policy/jacobian_erank_sample_mean",
            "investigator/policy/jacobian_erank_sample_std",
            "investigator/actor/dead_neuron_frac",
            "investigator/actor/feature_norm",
            "investigator/policy/action_mean_variance",
        ]:
            vals = self._history.get(key, [])
            if vals:
                all_v = [v for _, v in vals]
                summary[f"{key}/final"] = all_v[-1]
                summary[f"{key}/min"] = min(all_v)
                summary[f"{key}/max"] = max(all_v)
                summary[f"{key}/mean"] = float(np.mean(all_v))

        dead = [v for _, v in self._history.get(
            "investigator/actor/dead_neuron_frac", [])]
        summary["investigator/collapse_detected"] = any(v > 0.5 for v in dead)
        wandb.run.summary.update(summary)

    # ---------------------------------------------------------
    # Live metrics (every log_interval)
    # ---------------------------------------------------------

    def _compute_live_metrics(self, iteration: int):
        actor = self._get_actor()
        obs = self.runner.alg.policy.get_actor_obs(self.runner.env.get_observations()).to(self.device)
        obs = self._normalize_actor_obs(obs)
        metrics = {}

        with torch.no_grad():
            # -- Actor features --
            actor_feats = self._extract_head_input(actor, obs)
            if actor_feats is not None:
                _, S_a, _ = torch.linalg.svd(actor_feats, full_matrices=False)
                metrics["investigator/actor/feature_erank"] = \
                    effective_rank(S_a, self.cfg.erank_eps)
                metrics["investigator/actor/pca_rank_99"] = pca_rank(S_a, 0.99)

                neuron_stds = actor_feats.std(dim=0)
                dead = (neuron_stds < self.cfg.dead_neuron_threshold).sum().item()
                total = actor_feats.shape[1]
                metrics["investigator/actor/dead_neurons"] = dead
                metrics["investigator/actor/dead_neuron_frac"] = dead / max(total, 1)
                metrics["investigator/actor/neuron_std_median"] = \
                    neuron_stds.median().item()

                # Backup raw arrays for post-hoc histogram reconstruction
                torch.save(S_a.cpu(), self._log_dir / f"spectra/actor_live_sv_{iteration}.pt")
                torch.save(neuron_stds.cpu(), self._log_dir / f"spectra/neuron_std_{iteration}.pt")

                # [WANDB INTEGRATION] live actor SV + neuron-std histograms
                if self.cfg.wandb_spectral_histograms and _WANDB_AVAILABLE:
                    S_norm = (S_a / S_a.sum()).cpu().numpy()
                    metrics["investigator/spectra/actor_features"] = \
                        wandb.Histogram(S_norm)
                    metrics["investigator/actor/neuron_std_dist"] = \
                        wandb.Histogram(neuron_stds.cpu().numpy())

                # -- Representation norm (policy-head input) --
                norms = actor_feats.norm(dim=-1)
                metrics["investigator/actor/feature_norm"] = norms.mean().item()
                metrics["investigator/actor/feature_norm_std"] = norms.std().item()

            # -- Policy variance --
            action_means = self._get_action_means(actor, obs)
            if action_means is not None:
                action_var_per_dim = action_means.var(dim=0)
                metrics["investigator/policy/action_mean_variance"] = \
                    action_var_per_dim.mean().item()
                metrics["investigator/policy/action_mean_variance_std"] = \
                    action_var_per_dim.std().item()

        self._log_metrics(metrics, iteration)

    # ---------------------------------------------------------
    # Weight ranks (every weight_rank_interval)
    # ---------------------------------------------------------

    def _compute_weight_ranks(self, iteration: int):
        metrics = {}
        layer_eranks_actor = []

        with torch.no_grad():
            for label, model in [("actor", self._get_actor())]:
                for i, (name, layer) in enumerate(_find_linear_layers(model)):
                    _, S, _ = torch.linalg.svd(
                        layer.weight.data, full_matrices=False)
                    er = effective_rank(S, self.cfg.erank_eps)
                    metrics[f"investigator/weights/{label}/layer_{i}_erank"] = er

                    # Backup raw SVs for post-hoc histogram reconstruction
                    torch.save(S.cpu(), self._log_dir / f"spectra/weight_{label}_L{i}_sv_{iteration}.pt")

                    if label == "actor":
                        layer_eranks_actor.append((f"L{i}", er))

                    # [WANDB INTEGRATION] per-layer weight SV histogram
                    if self.cfg.wandb_spectral_histograms and _WANDB_AVAILABLE:
                        metrics[f"investigator/spectra/weight_{label}_L{i}"] = \
                            wandb.Histogram(S.cpu().numpy())

        self._log_metrics(metrics, iteration)

    # ---------------------------------------------------------
    # Checkpoint metrics (expensive, every checkpoint_interval)
    # ---------------------------------------------------------

    def _compute_checkpoint_metrics(self, iteration: int):
        actor = self._get_actor()
        eval_obs = self._normalize_actor_obs(self._fixed_eval_obs.to(self.device))
        metrics = {}

        # -- Gram matrix + feature erank on fixed eval obs --
        with torch.no_grad():
            features = self._extract_head_input(actor, eval_obs)
            if features is not None:
                # Feature erank on fixed eval obs
                _, S_fixed, V_fixed = torch.linalg.svd(
                    features.cpu(), full_matrices=False)
                feat_er_fixed = effective_rank(S_fixed, self.cfg.erank_eps)
                metrics["investigator/actor/feature_erank_fixed"] = feat_er_fixed

                gram = compute_gram_cosine(features)
                _, S_gram, _ = torch.linalg.svd(gram)
                gram_er = effective_rank(S_gram, self.cfg.erank_eps)
                metrics["investigator/gram/erank"] = gram_er

                # Local save
                if self.cfg.save_grams:
                    torch.save(gram.cpu(),
                               self._log_dir / f"grams/gram_{iteration}.pt")
                if self.cfg.save_features:
                    torch.save(features.cpu(),
                               self._log_dir / f"features/features_{iteration}.pt")

                # [WANDB INTEGRATION] Gram SV histogram
                if self.cfg.wandb_spectral_histograms and _WANDB_AVAILABLE:
                    S_n = (S_gram / S_gram.sum()).cpu().numpy()
                    metrics["investigator/spectra/gram_sv"] = \
                        wandb.Histogram(S_n)

                if self.cfg.gram_heatmap:
                    self._emit_figure(
                        self._render_gram_figure(gram.cpu(), iteration, gram_er),
                        "gram", iteration, metrics, "investigator/gram/heatmap")

                if self.cfg.feature_heatmap:
                    self._emit_figure(
                        self._render_feature_heatmap(
                            features.cpu(), V_fixed, iteration, feat_er_fixed),
                        "features", iteration, metrics, "investigator/features/heatmap")

        # -- Jacobian rank --
        jac_obs = eval_obs[:self.cfg.jacobian_batch_size]
        jac_er, jac_S, jac_matrix, jac_std, jac_ps_eranks = \
            self._compute_jacobian_erank(actor, jac_obs)
        if jac_er is not None:
            metrics["investigator/policy/jacobian_erank"] = jac_er

            # Per-sample Jacobian erank stats
            if jac_ps_eranks is not None:
                metrics["investigator/policy/jacobian_erank_sample_mean"] = \
                    jac_ps_eranks.mean().item()
                metrics["investigator/policy/jacobian_erank_sample_std"] = \
                    jac_ps_eranks.std().item()
                # [WANDB INTEGRATION] per-sample Jacobian erank histogram
                if self.cfg.wandb_spectral_histograms and _WANDB_AVAILABLE:
                    metrics["investigator/spectra/jacobian_persample_eranks"] = \
                        wandb.Histogram(jac_ps_eranks.numpy())

            if self.cfg.save_jacobian_spectra and jac_S is not None:
                torch.save(jac_S.cpu(),
                           self._log_dir / f"jacobians/jac_sv_{iteration}.pt")
            if self.cfg.save_jacobian_spectra and jac_ps_eranks is not None:
                torch.save(jac_ps_eranks,
                           self._log_dir / f"jacobians/jac_persample_eranks_{iteration}.pt")
            if self.cfg.save_jacobians and jac_matrix is not None:
                torch.save(jac_matrix.cpu(),
                           self._log_dir / f"jacobians/jac_matrix_{iteration}.pt")
            if self.cfg.save_jacobians and jac_std is not None:
                torch.save(jac_std.cpu(),
                           self._log_dir / f"jacobians/jac_std_{iteration}.pt")

            # [WANDB INTEGRATION] Jacobian mean SV histogram
            if (self.cfg.wandb_spectral_histograms and _WANDB_AVAILABLE
                    and jac_S is not None):
                metrics["investigator/spectra/jacobian_sv"] = \
                    wandb.Histogram(jac_S.cpu().numpy())

            if self.cfg.jacobian_heatmap and jac_matrix is not None:
                self._emit_figure(
                    self._render_jacobian_heatmap(jac_matrix.cpu(), iteration, jac_er),
                    "jacobian", iteration, metrics, "investigator/jacobian/heatmap")
            if self.cfg.jacobian_heatmap and jac_std is not None:
                self._emit_figure(
                    self._render_jacobian_std_heatmap(jac_std.cpu(), iteration),
                    "jacobian_std", iteration, metrics, "investigator/jacobian/std_heatmap")

        # Phase labels and observations must come from the same live environment state.
        if self.cfg.phase_log_enabled and self._phase_fn is not None:
            try:
                self._compute_phase_metrics(iteration, metrics)
            except Exception as e:
                warnings.warn(f"[Investigator] Phase metrics failed: {e}")

        if self.cfg.weight_heatmap:
            self._emit_figure(
                self._render_weight_heatmap(actor, iteration),
                "weights", iteration, metrics, "investigator/weights/heatmap")

        # -- Full weight spectra (disk only) --
        if self.cfg.save_spectra:
            self._save_weight_spectra(actor, iteration, "actor")
            # Backup raw weight matrices for post-hoc weight-heatmap reconstruction
            weights = {f"layer_{i}_{name}": layer.weight.data.cpu()
                       for i, (name, layer) in enumerate(_find_linear_layers(actor))}
            torch.save(weights, self._log_dir / f"spectra/actor_weights_{iteration}.pt")

        self._log_metrics(metrics, iteration)

    def _compute_phase_metrics(self, iteration: int, metrics: dict) -> None:
        """Measure ranks on live observations grouped by simultaneous foot contacts."""
        raw_obs = self.runner.alg.policy.get_actor_obs(
            self.runner.env.get_observations()
        ).to(self.device)
        phases = self._phase_fn(self.runner.env).to(self.device).long()
        if phases.ndim != 1 or phases.shape[0] != raw_obs.shape[0]:
            raise ValueError("Contact counts must have one value per live observation.")
        if ((phases < 0) | (phases > self._phase_contact_count)).any():
            raise ValueError("Contact count exceeds the number of tracked feet.")

        obs = self._normalize_actor_obs(raw_obs.detach())
        actor = self._get_actor()
        n_feet = self._phase_contact_count
        for count, label in enumerate(CONTACT_COUNT_LABELS[n_feet]):
            metrics[f"investigator_phase/dist/frac_{label}"] = (
                (phases == count).float().mean().item()
            )
        metrics["investigator/robot/n_dof"] = float(
            _find_linear_layers(actor)[-1][1].out_features
        )

        total_steps = iteration * self.runner.num_steps_per_env * self.runner.env.num_envs
        rows = []
        for alias, counts in PHASE_GROUPS[n_feet].items():
            mask = torch.ones_like(phases, dtype=torch.bool) if counts is None else torch.isin(
                phases, torch.tensor(counts, device=phases.device)
            )
            phase_obs = obs[mask]
            n_samples = phase_obs.shape[0]
            valid = int(n_samples >= self.cfg.min_samples_per_phase)
            metrics[f"investigator_phase/{alias}/sample_count"] = float(n_samples)
            metrics[f"investigator_phase/{alias}/valid_rank_eval"] = float(valid)
            row = {field: float("nan") for field in PHASE_CSV_FIELDS}
            row.update(global_step=total_steps, phase=alias, sample_count=n_samples, valid=valid)

            if valid:
                with torch.no_grad():
                    features = self._extract_head_input(actor, phase_obs)
                    if features is not None:
                        singular_values = torch.linalg.svdvals(features)
                        feature_erank = effective_rank(singular_values, self.cfg.erank_eps)
                        pca = pca_rank(singular_values)
                        metrics[f"investigator_phase/{alias}/actor/feature_erank"] = feature_erank
                        metrics[f"investigator_phase/{alias}/actor/pca_rank_99"] = float(pca)
                        row["feature_erank"] = feature_erank
                        row["pca_rank_99"] = pca

                    gram_features = self._extract_head_input(actor, phase_obs[:512])
                    if gram_features is not None:
                        gram = compute_gram_cosine(gram_features)
                        gram_erank = effective_rank(torch.linalg.svdvals(gram), self.cfg.erank_eps)
                        metrics[f"investigator_phase/{alias}/gram/erank"] = gram_erank
                        row["gram_erank"] = gram_erank

                jac_erank, _, _, _, per_sample = self._compute_jacobian_erank(
                    actor, phase_obs[:self.cfg.jacobian_batch_size]
                )
                if jac_erank is not None:
                    metrics[f"investigator_phase/{alias}/policy/jacobian_erank"] = jac_erank
                    dof_gap = metrics["investigator/robot/n_dof"] - jac_erank
                    metrics[f"investigator_phase/{alias}/policy/jacobian_dof_gap"] = dof_gap
                    row["jacobian_erank"] = jac_erank
                    row["jacobian_dof_gap"] = dof_gap
                if per_sample is not None:
                    mean = per_sample.mean().item()
                    std = per_sample.std().item()
                    metrics[f"investigator_phase/{alias}/policy/jacobian_local_erank_mean"] = mean
                    metrics[f"investigator_phase/{alias}/policy/jacobian_local_erank_std"] = std
                    row["jacobian_local_erank_mean"] = mean
                    row["jacobian_local_erank_std"] = std
            rows.append(row)

        csv_path = self._log_dir / "phase_rank_timeseries.csv"
        with open(csv_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=PHASE_CSV_FIELDS)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)

    # ---------------------------------------------------------
    # Feature extraction via hooks
    # ---------------------------------------------------------

    def _extract_head_input(
        self, model: nn.Module, obs: torch.Tensor
    ) -> torch.Tensor | None:
        """Representation fed to the policy head (input of the last Linear).

        MLP: last hidden activation. SimBa: post_ln(residual stream).
        """
        captured = {}
        try:
            _, head = _find_head_linear(model)
        except ValueError:
            return None

        def hook(module, inp):
            captured["f"] = inp[0].detach()

        h = head.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                model(obs)
        finally:
            h.remove()
        return captured.get("f")

    def _get_action_means(
        self, actor: nn.Module, obs: torch.Tensor
    ) -> torch.Tensor | None:
        """Get action means from the actor (nn.Sequential -> raw tensor)."""
        with torch.no_grad():
            try:
                return actor(obs).detach()
            except Exception:
                return None

    # ---------------------------------------------------------
    # Jacobian
    # ---------------------------------------------------------

    def _compute_jacobian_erank(
        self, actor: nn.Module, obs: torch.Tensor
    ) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None]:
        try:
            if self.cfg.jacobian_use_vmap and hasattr(torch, "func"):
                return self._jacobian_vmap(actor, obs)
            return self._jacobian_loop(actor, obs)
        except Exception as e:
            warnings.warn(f"[Investigator] Jacobian failed: {e}")
            return None, None, None, None, None

    def _jacobian_vmap(self, actor, obs):
        from torch.func import jacrev, vmap

        def fwd(x):
            return actor(x.unsqueeze(0)).squeeze(0)

        was_training = actor.training
        actor.eval()
        try:
            with torch.enable_grad():
                J = vmap(jacrev(fwd))(obs)  # [B, a_dim, o_dim]
        finally:
            actor.train(was_training)

        J_mean = J.mean(dim=0)
        J_std = J.std(dim=0)
        _, S, _ = torch.linalg.svd(J_mean, full_matrices=False)
        per_sample_eranks = torch.tensor([
            effective_rank(torch.linalg.svdvals(J[i]), self.cfg.erank_eps)
            for i in range(J.shape[0])
        ])
        return (effective_rank(S, self.cfg.erank_eps), S.detach().cpu(),
                J_mean.detach().cpu(), J_std.detach().cpu(), per_sample_eranks)

    def _jacobian_loop(self, actor, obs):
        obs_g = obs.detach().requires_grad_(True)
        was_training = actor.training
        actor.eval()
        try:
            out = actor(obs_g)
            a_dim = out.shape[-1]

            J = torch.zeros(obs.shape[0], a_dim, obs.shape[-1], device=obs.device)
            for a in range(a_dim):
                g = torch.zeros_like(out)
                g[:, a] = 1.0
                grads = torch.autograd.grad(
                    out, obs_g, g, retain_graph=(a < a_dim - 1))[0]
                J[:, a, :] = grads
        finally:
            actor.train(was_training)

        J_mean = J.mean(dim=0)
        J_std = J.std(dim=0)
        _, S, _ = torch.linalg.svd(J_mean, full_matrices=False)
        per_sample_eranks = torch.tensor([
            effective_rank(torch.linalg.svdvals(J[i]), self.cfg.erank_eps)
            for i in range(J.shape[0])
        ])
        return (effective_rank(S, self.cfg.erank_eps), S.detach().cpu(),
                J_mean.detach().cpu(), J_std.detach().cpu(), per_sample_eranks)

    # ---------------------------------------------------------
    # Weight spectra (disk)
    # ---------------------------------------------------------

    def _save_weight_spectra(self, model, iteration, label):
        spectra = {}
        with torch.no_grad():
            for i, (name, layer) in enumerate(_find_linear_layers(model)):
                _, S, _ = torch.linalg.svd(
                    layer.weight.data, full_matrices=False)
                spectra[f"layer_{i}_{name}"] = S.cpu()
        torch.save(
            spectra,
            self._log_dir / f"spectra/{label}_spectra_{iteration}.pt"
        )

    # ---------------------------------------------------------
    # Model access (rsl_rl ActorCritic: policy.actor / policy.critic)
    # ---------------------------------------------------------

    def _get_actor(self) -> nn.Module:
        return self.runner.alg.policy.actor

    def _normalize_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Feed metrics the same actor inputs used by RSL RL's policy."""
        normalizer = getattr(self.runner.alg.policy, "actor_obs_normalizer", None)
        if normalizer is None:
            return obs
        with torch.no_grad():
            return normalizer(obs)

    def _emit_figure(self, fig, name: str, iteration: int, metrics: dict, wandb_key: str):
        """Save a checkpoint figure (PNG + PDF) and attach it to the W&B row if a run exists."""
        if fig is None:
            return
        save_figure(fig, self._log_dir / "plots" / "checkpoints" / f"{name}_{iteration}")
        if self.cfg.wandb_log and _WANDB_AVAILABLE and wandb.run:
            metrics[wandb_key] = wandb.Image(figure_image(fig))
        plt.close(fig)

    def _action_labels(self, a_dim: int) -> list[str]:
        actions = self._layout.get("actions") or []
        return actions if len(actions) == a_dim else [f"$a_{{{i}}}$" for i in range(a_dim)]

    def _obs_groups(self, o_dim: int) -> list:
        groups = self._layout.get("obs_groups") or []
        return groups if groups and groups[-1][2] == o_dim else []

    def _sample_label(self) -> str:
        sort_label = self._layout.get("sort_label")
        return f"Sample (sorted by {sort_label})" if sort_label else "Sample"

    def _arch_color(self) -> str:
        return PALETTE["simba" if self._layout.get("arch") == "SimBa" else "mlp"]

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    def _log_metrics(self, metrics: dict, iteration: int):
        scalars = {}
        wandb_objs = {}

        # [WANDB INTEGRATION] separate vendor (Histogram/Image) payloads from scalars
        for k, v in metrics.items():
            if _WANDB_AVAILABLE and isinstance(v, (wandb.Histogram, wandb.Image)):
                wandb_objs[k] = v
            elif _WANDB_AVAILABLE and hasattr(v, '__class__') and \
                    v.__class__.__module__.startswith('wandb'):
                wandb_objs[k] = v
            else:
                scalars[k] = v

        for k, v in scalars.items():
            if isinstance(v, (int, float)):
                self._history.setdefault(k, []).append((iteration, v))

        # [WANDB INTEGRATION] central wandb.log call (scalars + non-scalar payloads)
        if self.cfg.wandb_log and _WANDB_AVAILABLE and wandb.run:
            wandb.log(
                {**scalars, **wandb_objs, "train_iteration": iteration},
                step=iteration,
            )

    # ===================================================================
    # Checkpoint heatmaps
    # ===================================================================

    def _render_gram_figure(self, gram: torch.Tensor, iteration: int,
                            erank_val: float):
        """Cosine Gram of head-input features; negatives clipped to 0 (pattern view)."""
        g = gram.numpy()
        if self._sort_indices is not None:
            idx = self._sort_indices.numpy()
            g = g[np.ix_(idx, idx)]

        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.85), layout="constrained")
        im = ax.imshow(np.clip(g, 0.0, 1.0), cmap=CMAP_SEQUENTIAL, vmin=0.0, vmax=1.0,
                       interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(self._sample_label())
        ax.set_ylabel(self._sample_label())
        ax.set_title(f"Gram matrix, iteration {iteration} (erank {erank_val:.1f})")
        _colorbar(fig, im, ax, "Cosine similarity")
        return fig

    def _render_feature_heatmap(self, features: torch.Tensor,
                                V: torch.Tensor, iteration: int,
                                erank_val: float):
        """Samples x head-input units; rows in sort order, units by top right singular vector."""
        f = features.numpy()
        if self._sort_indices is not None:
            f = f[self._sort_indices.numpy()]
        f = f[:, np.argsort(-np.abs(V[0].numpy()))]

        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.85), layout="constrained")
        im = ax.imshow(f, cmap=CMAP_DIVERGING, norm=_symmetric_norm(f), aspect="auto",
                       interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"Head-input unit (sorted by $|v_1|$)")
        ax.set_ylabel(self._sample_label())
        ax.set_title(f"Features, iteration {iteration} (erank {erank_val:.1f})")
        _colorbar(fig, im, ax, "Activation")
        return fig

    def _jacobian_figure(self, J: np.ndarray, rows: int = 1):
        a_dim = J.shape[0]
        row_h = 0.105 * a_dim + 0.35
        return plt.subplots(rows, 1, figsize=(FULL_WIDTH, rows * row_h + 0.45),
                            squeeze=False, layout="constrained")

    def _label_jacobian(self, ax, a_dim: int, o_dim: int, sep_color: str,
                        show_x: bool = True):
        """Actions on y (joint labels), observation terms on x (horizontal symbols)."""
        actions = self._action_labels(a_dim)
        ax.set_yticks(range(a_dim))
        ax.set_yticklabels(actions)
        ax.set_ylabel("Action")
        groups = self._obs_groups(o_dim)
        if groups:
            ax.set_xticks([(s + e - 1) / 2 for _, s, e in groups])
            ax.set_xticklabels([n for n, _, _ in groups] if show_x else [])
            for _, s, _ in groups[1:]:
                ax.axvline(s - 0.5, color=sep_color, linewidth=0.5)
        elif not show_x:
            ax.set_xticklabels([])
        if show_x:
            ax.set_xlabel("Observation")
        # Separate limbs when actions come in runs sharing a prefix (FL, LF, ...).
        prefixes = [a.split(" ")[0] for a in actions]
        bounds = [i for i in range(1, a_dim) if prefixes[i] != prefixes[i - 1]]
        runs = np.diff([0, *bounds, a_dim])
        if bounds and runs.min() >= 2:
            for i in bounds:
                ax.axhline(i - 0.5, color=sep_color, linewidth=0.5)
        ax.tick_params(length=2.0, width=0.6)

    def _render_jacobian_heatmap(self, J_mean: torch.Tensor, iteration: int,
                                 erank_val: float):
        J = J_mean.numpy()
        fig, axes = self._jacobian_figure(J)
        ax = axes[0, 0]
        im = ax.imshow(J, cmap=CMAP_DIVERGING, norm=_symmetric_norm(J), aspect="auto",
                       interpolation="nearest")
        self._label_jacobian(ax, *J.shape, sep_color=PALETTE["ghost"])
        ax.set_title(f"Mean policy Jacobian, iteration {iteration} (erank {erank_val:.1f})")
        _colorbar(fig, im, ax, r"$\partial a_i / \partial o_j$")
        return fig

    def _render_jacobian_std_heatmap(self, J_std: torch.Tensor, iteration: int):
        J = J_std.numpy()
        fig, axes = self._jacobian_figure(J)
        ax = axes[0, 0]
        im = ax.imshow(J, cmap=CMAP_SEQUENTIAL, vmin=0.0, vmax=max(float(J.max()), 1e-12),
                       aspect="auto", interpolation="nearest")
        self._label_jacobian(ax, *J.shape, sep_color=PALETTE["ghost"])
        ax.set_title(f"Policy Jacobian std. across samples, iteration {iteration}")
        _colorbar(fig, im, ax, r"std. $\partial a_i / \partial o_j$")
        return fig

    def _render_weight_heatmap(self, model: nn.Module, iteration: int):
        """Weight matrices of all Linear layers, one panel per layer."""
        layers = _find_linear_layers(model)
        if not layers:
            return None
        n = len(layers)
        ncols = min(n, 4)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, nrows * 1.75),
                                 squeeze=False, layout="constrained")
        with torch.no_grad():
            for i, (_, layer) in enumerate(layers):
                W = layer.weight.data.cpu().numpy()
                er = effective_rank(torch.linalg.svdvals(layer.weight.data), self.cfg.erank_eps)
                ax = axes[i // ncols, i % ncols]
                im = ax.imshow(W, cmap=CMAP_DIVERGING, norm=_symmetric_norm(W), aspect="auto",
                               interpolation="antialiased")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(rf"$W_{{{i}}}$, {W.shape[0]}$\times${W.shape[1]}"
                             f"\nerank {er:.1f}")
                _colorbar(fig, im, ax, None)
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].set_axis_off()
        axes[0, 0].set_ylabel("Output")
        axes[-1, 0].set_xlabel("Input")
        fig.suptitle(f"Actor weights, iteration {iteration}", fontsize=9.0)
        return fig

    # ===================================================================
    # Post-hoc report generation
    # ===================================================================

    def generate_report(self, compare_dirs: list[str | Path] | None = None):
        """Generate the paper-style report figures (PNG + PDF) from saved data."""

        plots_dir = self._log_dir / "plots"
        runs = [(self._log_dir.parent.name, self._layout.get("arch", "MLP"),
                 self._load_history(self._log_dir))]
        for d in compare_dirs or []:
            d = Path(d)
            label = d.parent.name
            runs.append((label, _read_layout(d).get("arch") or arch_of(label),
                         self._load_history(d)))

        self._plot_rank_evolution(runs, plots_dir)
        self._plot_gram_evolution_grid(plots_dir)
        self._plot_feature_heatmap_grid(plots_dir)
        self._plot_jacobian_heatmap_grid(plots_dir)
        self._plot_spectral_evolution(plots_dir)
        self._plot_collapse_indicators(runs, plots_dir)
        print(f"[Investigator] Report -> {plots_dir}")

    def _load_history(self, inv_dir: Path) -> dict:
        p = inv_dir / "history.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return self._history

    def _plot_rank_evolution(self, runs, out):
        keys = [
            ("investigator/actor/feature_erank", "Feature"),
            ("investigator/policy/jacobian_erank", "Jacobian"),
            ("investigator/gram/erank", "Gram"),
        ]
        fig = _plot_history_panels(runs, keys, "Effective rank")
        save_figure(fig, out / "rank_evolution")
        plt.close(fig)

    def _plot_collapse_indicators(self, runs, out):
        keys = [
            ("investigator/actor/feature_norm", "Head-input L2 norm"),
            ("investigator/actor/dead_neuron_frac", "Dead-unit fraction"),
            ("investigator/policy/action_mean_variance", "Policy variance"),
        ]
        fig = _plot_history_panels(runs, keys, None)
        save_figure(fig, out / "collapse_indicators")
        plt.close(fig)

    def _plot_gram_evolution_grid(self, out):
        gfiles = _checkpoint_files(self._log_dir / "grams", "gram_*.pt", 6)
        if not gfiles:
            return
        n = len(gfiles)
        side = FULL_WIDTH / n
        fig, axes = plt.subplots(3, n, figsize=(FULL_WIDTH, 2 * side + 1.0),
                                 height_ratios=[side, side, 1.0], squeeze=False,
                                 layout="constrained")
        color = self._arch_color()
        for col, (it, gf) in enumerate(gfiles):
            gram = torch.load(gf, weights_only=True)
            g = gram.numpy()
            if self._sort_indices is not None:
                idx = self._sort_indices.numpy()
                g = g[np.ix_(idx, idx)]
            Sg = torch.linalg.svdvals(gram)
            er = effective_rank(Sg, self.cfg.erank_eps)

            im_signed = axes[0, col].imshow(g, cmap=CMAP_DIVERGING, vmin=-1.0, vmax=1.0,
                                            interpolation="antialiased")
            axes[0, col].set_title(f"Iteration {it}\nerank {er:.1f}")
            im_clip = axes[1, col].imshow(np.clip(g, 0.0, 1.0), cmap=CMAP_SEQUENTIAL,
                                          vmin=0.0, vmax=1.0, interpolation="antialiased")
            for r in (0, 1):
                axes[r, col].set_xticks([])
                axes[r, col].set_yticks([])

            ax_s = axes[2, col]
            Sn = (Sg / Sg.sum()).numpy()
            ax_s.semilogy(Sn, color=color, linewidth=0.9, zorder=3)
            ax_s.axhline(1 / len(Sn), color=PALETTE["ghost"], linestyle="--", linewidth=0.6,
                         zorder=2)
            style_axes(ax_s)
            ax_s.set_xlabel("Index")
            if col:
                ax_s.tick_params(labelleft=False)
        axes[0, 0].set_ylabel("Signed")
        axes[1, 0].set_ylabel("Clipped at 0")
        axes[2, 0].set_ylabel(r"Normalized $\sigma_i$")
        _colorbar(fig, im_signed, axes[0, :].tolist(), "Cosine similarity")
        _colorbar(fig, im_clip, axes[1, :].tolist(), "Cosine similarity")
        save_figure(fig, out / "gram_evolution")
        plt.close(fig)

    def _plot_feature_heatmap_grid(self, out):
        ffiles = _checkpoint_files(self._log_dir / "features", "features_*.pt", 4)
        if not ffiles:
            return
        panels = []
        for it, ff in ffiles:
            features = torch.load(ff, weights_only=True)
            f = features.numpy()
            if self._sort_indices is not None:
                f = f[self._sort_indices.numpy()]
            _, S, V = torch.linalg.svd(features, full_matrices=False)
            panels.append((it, f[:, np.argsort(-np.abs(V[0].numpy()))],
                           effective_rank(S, self.cfg.erank_eps)))
        norm = _symmetric_norm(np.concatenate([p[1].ravel() for p in panels]))

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(FULL_WIDTH, 2.2), squeeze=False,
                                 layout="constrained")
        for col, (it, f, er) in enumerate(panels):
            ax = axes[0, col]
            im = ax.imshow(f, cmap=CMAP_DIVERGING, norm=norm, aspect="auto",
                           interpolation="antialiased")
            ax.set_title(f"Iteration {it}\nerank {er:.1f}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Head-input unit")
        axes[0, 0].set_ylabel(self._sample_label())
        _colorbar(fig, im, axes[0, :].tolist(), "Activation")
        save_figure(fig, out / "feature_evolution")
        plt.close(fig)

    def _plot_jacobian_heatmap_grid(self, out):
        jfiles = _checkpoint_files(self._log_dir / "jacobians", "jac_matrix_*.pt", 4)
        if not jfiles:
            return
        mats = [(it, torch.load(jf, weights_only=True)) for it, jf in jfiles]
        norm = _symmetric_norm(np.concatenate([J.numpy().ravel() for _, J in mats]))
        a_dim, o_dim = mats[0][1].shape

        fig, axes = self._jacobian_figure(mats[0][1].numpy(), rows=len(mats))
        for row, (it, J) in enumerate(mats):
            ax = axes[row, 0]
            er = effective_rank(torch.linalg.svdvals(J), self.cfg.erank_eps)
            im = ax.imshow(J.numpy(), cmap=CMAP_DIVERGING, norm=norm, aspect="auto",
                           interpolation="nearest")
            self._label_jacobian(ax, a_dim, o_dim, PALETTE["ghost"],
                                 show_x=row == len(mats) - 1)
            ax.set_title(f"Iteration {it} (erank {er:.1f})", loc="left")
        _colorbar(fig, im, axes[:, 0].tolist(), r"$\partial a_i / \partial o_j$")
        save_figure(fig, out / "jacobian_evolution")
        plt.close(fig)

    def _plot_spectral_evolution(self, out):
        sfiles = _checkpoint_files(self._log_dir / "spectra", "actor_spectra_*.pt", 4)
        if not sfiles:
            return
        spectra = [(it, torch.load(sf, weights_only=True)) for it, sf in sfiles]
        lnames = list(spectra[0][1].keys())
        nl, n = len(lnames), len(spectra)
        fig, axes = plt.subplots(nl, n, figsize=(FULL_WIDTH, nl * 0.95 + 0.3),
                                 sharex="row", sharey="row", squeeze=False,
                                 layout="constrained")
        color = self._arch_color()
        for col, (it, sp) in enumerate(spectra):
            for row, ln in enumerate(lnames):
                S = sp[ln].numpy()
                S_n = S / S.max()
                ax = axes[row, col]
                ax.semilogy(S_n, linewidth=0.9, color=color, zorder=3)
                ax.fill_between(range(len(S_n)), S_n, alpha=0.15, color=color, zorder=2)
                style_axes(ax)
                er = effective_rank(torch.tensor(S), self.cfg.erank_eps)
                ax.text(0.95, 0.92, f"erank {er:.1f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5)
                if row == 0:
                    ax.set_title(f"Iteration {it}")
                if col == 0:
                    ax.set_ylabel(rf"$W_{{{row}}}$  $\sigma_i/\sigma_1$")
                if row == nl - 1:
                    ax.set_xlabel("Index")
        save_figure(fig, out / "spectral_evolution")
        plt.close(fig)


# ===================================================================
# Plot helpers
# ===================================================================

def _symmetric_norm(a: np.ndarray):
    vabs = float(np.abs(a).max()) if a.size else 0.0
    return TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs) if vabs > 0 else None


def _colorbar(fig, im, ax, label: str | None):
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, aspect=25)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(PALETTE["edge"])
    cb.ax.tick_params(length=2.0, width=0.5)
    if label:
        cb.set_label(label)
    return cb


def _checkpoint_files(directory: Path, pattern: str, max_n: int) -> list[tuple[int, Path]]:
    """Up to max_n evenly spaced (iteration, path) pairs, oldest first."""
    files = sorted(((int(p.stem.rsplit("_", 1)[1]), p) for p in directory.glob(pattern)))
    if not files:
        return []
    n = min(max_n, len(files))
    return [files[i] for i in np.linspace(0, len(files) - 1, n, dtype=int)]


def _read_layout(inv_dir: Path) -> dict:
    p = Path(inv_dir) / "layout.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _plot_history_panels(runs, keys, ylabel: str | None):
    """One panel per history key, one line per run (architecture color, paper name)."""
    styles = run_styles([(label, arch) for label, arch, _ in runs])
    fig, axes = plt.subplots(1, len(keys), figsize=(FULL_WIDTH, 1.9), squeeze=False,
                             layout="constrained")
    for ax, (k, title) in zip(axes[0], keys):
        for label, _, h in runs:
            if h.get(k):
                its, vs = zip(*h[k])
                ax.plot(its, vs, label=paper_name(label), linewidth=1.0, zorder=3,
                        marker="o" if len(its) < 8 else None, markersize=3.0,
                        markeredgecolor=PALETTE["edge"], markeredgewidth=0.4, **styles[label])
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
        style_axes(ax)
    if ylabel:
        axes[0, 0].set_ylabel(ylabel)
    if len(runs) > 1:
        axes[0, 0].legend(loc="best")
    return fig


# ===================================================================
# Cross-experiment comparison utility
# ===================================================================

def compare_experiments(
    experiment_dirs: dict[str, str | Path],
    output_dir: str | Path = "comparison_plots",
):
    """Compare investigator outputs across experiments.

    Args:
        experiment_dirs: {label: path_to_investigator_dir}; labels may be W&B config ids,
            which are shown with their paper names.
    """

    out = Path(output_dir)
    runs = []
    for label, d in experiment_dirs.items():
        hp = Path(d) / "history.json"
        if hp.exists():
            runs.append((label, _read_layout(d).get("arch") or arch_of(label),
                         json.loads(hp.read_text())))
    if not runs:
        return

    fig = _plot_history_panels(
        runs, [("investigator/actor/feature_erank", "Feature")], "Effective rank")
    fig.set_size_inches(COLUMN_WIDTH, 2.2)
    save_figure(fig, out / "cmp_feature_erank")
    plt.close(fig)

    styles = run_styles([(label, arch) for label, arch, _ in runs])
    ne = len(experiment_dirs)
    side = FULL_WIDTH / max(ne, 2)
    fig, axes = plt.subplots(2, ne, figsize=(FULL_WIDTH, side + 1.1),
                             height_ratios=[side, 1.0], squeeze=False, layout="constrained")
    im = None
    for col, (label, d) in enumerate(experiment_dirs.items()):
        gfiles = _checkpoint_files(Path(d) / "grams", "gram_*.pt", 1 << 30)
        if not gfiles or label not in styles:
            axes[0, col].set_axis_off()
            axes[1, col].set_axis_off()
            continue
        it, gf = gfiles[-1]
        gram = torch.load(gf, weights_only=True)
        Sg = torch.linalg.svdvals(gram)
        er = effective_rank(Sg)

        ax_g = axes[0, col]
        im = ax_g.imshow(np.clip(gram.numpy(), 0.0, 1.0), cmap=CMAP_SEQUENTIAL, vmin=0.0, vmax=1.0,
                         interpolation="antialiased")
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        ax_g.set_title(f"{paper_name(label)}\niteration {it}, erank {er:.1f}")

        ax_s = axes[1, col]
        Sn = (Sg / Sg.sum()).numpy()
        ax_s.semilogy(Sn, color=styles[label]["color"], linewidth=0.9, zorder=3)
        ax_s.axhline(1 / len(Sn), color=PALETTE["ghost"], linestyle="--", linewidth=0.6,
                     zorder=2)
        style_axes(ax_s)
        ax_s.set_xlabel("Index")
        if col == 0:
            ax_s.set_ylabel(r"Normalized $\sigma_i$")
    if im is not None:
        _colorbar(fig, im, axes[0, :].tolist(), "Cosine similarity")
    save_figure(fig, out / "cmp_grams")
    plt.close(fig)
    print(f"Comparison -> {out}")


# ===================================================================
# CLI for post-hoc
# ===================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--compare", nargs="*", default=None)
    parser.add_argument("--cross", nargs="*", default=None,
                        help="label:path pairs for cross-experiment")
    args = parser.parse_args()

    if args.cross:
        dirs = {}
        for item in args.cross:
            l, p = item.split(":", 1) if ":" in item \
                else (Path(item).parent.name, item)
            dirs[l] = p
        compare_experiments(dirs)
    else:
        inv = Investigator.__new__(Investigator)
        inv.cfg = InvestigatorCfg()
        inv._log_dir = Path(args.log_dir)
        inv._history = {}
        inv._layout = _read_layout(inv._log_dir)
        sort_p = inv._log_dir / "sort_indices.pt"
        inv._sort_indices = torch.load(sort_p, weights_only=True) if sort_p.exists() else None
        inv.generate_report(compare_dirs=args.compare)
