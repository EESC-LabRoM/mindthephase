# Copyright (c) 2024-2025, RAI Institute.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fetch W&B training data as a sparse LLM-friendly summary.

Usage:
    python scripts/rsl_rl/wandb_loader.py --run <display_name> [--slices 5] [--human]
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env")

import argparse

import numpy as np
import wandb

# ── Edit this dict to choose which metrics to fetch ─────────────────────────
METRICS = {
    # Training performance
    "mean_reward":      "investigator/train/mean_reward",
    "episode_length":   "investigator/train/mean_episode_length",
    "value_loss":       "investigator/train/value_loss",
    "entropy":          "investigator/train/entropy",
    # Representation health
    "feature_erank":    "investigator/actor/feature_erank",
    "gram_erank":       "investigator/gram/erank",
    "jacobian_erank":   "investigator/policy/jacobian_erank",
    # Metrics
    "vel_yaw_error": "metrics/base_velocity/error_vel_yaw",
    "vel_xy_error":  "metrics/base_velocity/error_vel_xy",
    "policy_noise": "policy/mean_noise_std",
}

ENTITY = "tommaselli"
PROJECT = "mymind"


def fetch_history(run, metrics: dict) -> dict[str, list[tuple[int, float]]]:
    data: dict[str, list] = {}
    for name, key in metrics.items():
        rows = []
        for row in run.scan_history(keys=[key, "_step"]):
            if key in row and row[key] is not None:
                rows.append((row.get("_step", len(rows)), float(row[key])))
        data[name] = rows
    return data


def compute_slices(data: dict, n_slices: int) -> list[dict]:
    all_steps = [s for series in data.values() for s, _ in series]
    if not all_steps:
        return []
    step_min, step_max = min(all_steps), max(all_steps)
    window = (step_max - step_min) / n_slices
    slices = []
    for i in range(n_slices):
        lo = step_min + i * window
        hi = step_min + (i + 1) * window
        metrics_out = {}
        for name, series in data.items():
            vals = [v for s, v in series if lo <= s < hi]
            metrics_out[name] = (np.mean(vals), np.std(vals)) if vals else None
        slices.append({"range": (int(lo), int(hi)), "metrics": metrics_out})
    return slices


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    mean, std = val
    return f"{mean:.3g}±{std:.3g}"


def format_llm(run_name: str, total_steps: int | str, slices: list) -> str:
    window = slices[0]["range"][1] - slices[0]["range"][0] if slices else 0
    lines = [
        f"RUN: {run_name} | STEPS: {total_steps} | SLICES: {len(slices)} | WINDOW: {window}",
        "",
    ]
    for i, s in enumerate(slices, 1):
        lo, hi = s["range"]
        parts = " ".join(f"{k}={_fmt(v)}" for k, v in s["metrics"].items())
        lines.append(f"[S{i}:{lo}-{hi}] {parts}")
    return "\n".join(lines)


def format_human(run_name: str, total_steps: int | str, slices: list) -> str:
    window = slices[0]["range"][1] - slices[0]["range"][0] if slices else 0
    lines = [
        f"=== {run_name} ===",
        f"Steps: {total_steps}  |  Slices: {len(slices)}  |  Window: {window} steps",
    ]
    for i, s in enumerate(slices, 1):
        lo, hi = s["range"]
        lines.append(f"\n--- Slice {i}  ({lo} – {hi}) ---")
        for k, v in s["metrics"].items():
            lines.append(f"  {k:<20} {_fmt(v)}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run", default=None, help="W&B run display name (default: latest run in project)")
    p.add_argument("--slices", type=int, default=5, help="Number of step windows (default: 5)")
    p.add_argument("--human", action="store_true", default=False, help="Human-readable output")
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    args = p.parse_args()

    api = wandb.Api()
    if args.run is None:
        runs = list(api.runs(f"{args.entity}/{args.project}", order="-created_at", per_page=1))
        if not runs:
            print(f"ERROR: no runs in {args.entity}/{args.project}")
            return
        run = runs[0]
        args.run = run.name
    else:
        runs = list(api.runs(f"{args.entity}/{args.project}", filters={"display_name": args.run}))
        if not runs:
            print(f"ERROR: run '{args.run}' not found in {args.entity}/{args.project}")
            return
        run = runs[0]

    total_steps = run.summary.get("_step", "?")
    print(f"[wandb_loader] fetching {len(METRICS)} metrics for '{args.run}' ({run.id})…", flush=True)

    data = fetch_history(run, METRICS)
    slices = compute_slices(data, args.slices)
    if not slices:
        print("ERROR: no data found for requested metrics")
        return

    output = format_human(args.run, total_steps, slices) if args.human else format_llm(args.run, total_steps, slices)
    print(output)


if __name__ == "__main__":
    main()
