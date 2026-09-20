"""
comparison.py -- pandas-based post-hoc comparison of training runs.

Loads per-run `investigator/history.json` files produced by investigator.py
into a long-format DataFrame, and draws small-multiples line plots for
side-by-side patch testing.

Run folder convention (matches train.py): logs/rsl_rl/{run_name}/.

Usage
-----

CLI:
    python scripts/rsl_rl/comparison.py --runs patch_a patch_b
    python scripts/rsl_rl/comparison.py --runs patch_a patch_b \
        --metrics investigator/gram/erank investigator/train/mean_reward \
        --out /tmp/cmp.png

Notebook:
    from comparison import load_runs_df, plot_metrics_grid, list_runs
    df = load_runs_df(["patch_a", "patch_b"])
    plot_metrics_grid(df, ["investigator/gram/erank",
                           "investigator/train/mean_reward"])
"""

# TODO(future): old interrupted runs may have raw investigator .pt files but
# no history.json, because history was originally flushed only at training end.
# New runs dump history incrementally; old partial runs need backfill from
# wandb/TensorBoard/raw .pt files if they must be compared.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
import re

from matplotlib.collections import LineCollection
import pandas as pd


DEFAULT_LOG_ROOT = Path("logs/rsl_rl")

DEFAULT_METRICS = [
    "investigator/train/mean_reward",
    "investigator/train/mean_episode_length",
    "investigator/train/value_loss",
    "investigator/train/surrogate_loss",
    "investigator/actor/feature_erank",
    "investigator/gram/erank",
    "investigator/policy/jacobian_erank",
    "investigator/train/collection_time",
]


def list_runs(log_root: str | Path = DEFAULT_LOG_ROOT) -> list[str]:
    """Return run names (subdirs of log_root) that contain history.json."""
    root = Path(log_root)
    if not root.exists():
        return []
    runs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "investigator" / "history.json").exists():
            runs.append(d.name)
    return runs


def load_runs_df(
    run_names: list[str] | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
) -> pd.DataFrame:
    """Load history.json from multiple runs into a long-format DataFrame.

    Columns: run, metric, iteration, value.
    If run_names is None, auto-discovers all runs under log_root.
    """
    root = Path(log_root)
    if run_names is None:
        run_names = list_runs(root)
    if not run_names:
        return pd.DataFrame(columns=["run", "metric", "iteration", "value"])

    frames = []
    for name in run_names:
        hist_path = root / name / "investigator" / "history.json"
        if not hist_path.exists():
            print(f"[comparison] skip '{name}' -- no {hist_path}")
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        rows = []
        for metric, series in hist.items():
            for it, val in series:
                rows.append((name, metric, int(it), float(val)))
        if rows:
            frames.append(pd.DataFrame(
                rows, columns=["run", "metric", "iteration", "value"]))

    if not frames:
        return pd.DataFrame(columns=["run", "metric", "iteration", "value"])
    return pd.concat(frames, ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, ax=None,
                log_y: bool = False):
    """Plot a single metric, one line per run."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    sub = df[df["metric"] == metric]
    if sub.empty:
        ax.set_title(f"{metric}\n(no data)", fontsize=8)
        ax.set_axis_off()
        return ax
    for run_name, g in sub.groupby("run"):
        g = g.sort_values("iteration")
        ax.plot(g["iteration"], g["value"], label=run_name, linewidth=1.3)
    ax.set_title(metric, fontsize=8)
    ax.set_xlabel("iteration", fontsize=7)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(labelsize=6)
    return ax


def plot_metrics_grid(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    ncols: int = 3,
    save_path: str | Path | None = None,
    suptitle: str | None = None,
):
    """Small-multiples grid: one subplot per metric, one line per run."""
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in set(df["metric"])]
        if not metrics:
            metrics = sorted(df["metric"].unique())[:9]
    n = len(metrics)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * 2.8),
                             squeeze=False)
    for i, metric in enumerate(metrics):
        ax = axes[i // ncols, i % ncols]
        plot_metric(df, metric, ax=ax)
    # hide unused cells
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_axis_off()

    # single shared legend (order stable across subplots)
    runs = sorted(df["run"].unique())
    handles = [plt.Line2D([], [], label=r, linewidth=1.3) for r in runs]
    # recolor handles to match first axis's lines if possible
    first_ax_lines = {l.get_label(): l for l in axes[0, 0].get_lines()}
    for h in handles:
        src = first_ax_lines.get(h.get_label())
        if src is not None:
            h.set_color(src.get_color())
    fig.legend(handles=handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(0.995, 0.995))

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[comparison] saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Rank × Performance analysis  (--rank-analysis)
# ---------------------------------------------------------------------------

_DEFAULT_RANK_METRICS = [
    "investigator/actor/feature_erank",
    "investigator/gram/erank",
    "investigator/policy/jacobian_erank",
]


def _load_api_key_from_cfg() -> str | None:
    """Fallback: read WANDB_API_KEY from rsl_rl_cfg.py."""
    cfg = (Path(__file__).resolve().parent.parent.parent
           / "source/mind/mind/tasks/locomotion"
             "/config/spot/agents/rsl_rl_cfg.py")
    if not cfg.exists():
        return None
    m = re.search(r'WANDB_API_KEY.*?["\']([a-f0-9]+)["\']', cfg.read_text())
    return m.group(1) if m else None


def _clean_metric_df(
    df: pd.DataFrame, run_name: str, metric: str,
) -> pd.DataFrame:
    """Return numeric, sorted metric data or raise on unusable input."""
    required = {"iteration", "value"}
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"wandb run '{run_name}' metric '{metric}' is missing columns: "
            f"{missing_cols}")

    result = df[["iteration", "value"]].copy()
    result["iteration"] = pd.to_numeric(result["iteration"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["iteration", "value"])
    if result.empty:
        raise ValueError(
            f"wandb run '{run_name}' has no usable rows for metric '{metric}'")
    return result.sort_values("iteration")


def fetch_wandb_perf(
    run_name: str,
    metric: str = "Train/mean_reward",
    project: str = "mymind",
    entity: str = "tommaselli",
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a scalar series from wandb.  Returns DataFrame[iteration, value].

    Caches to CSV in *cache_dir* so re-plotting doesn't re-download.
    """
    slug = metric.replace("/", "_").replace(" ", "_")
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"_wandb_cache_{slug}.csv"
        if cache_path.exists() and not refresh:
            print(f"[rank-analysis] cache hit: {cache_path}")
            try:
                cached = pd.read_csv(cache_path)
            except pd.errors.EmptyDataError as e:
                raise ValueError(
                    f"wandb cache for run '{run_name}' metric '{metric}' is "
                    f"empty: {cache_path}") from e
            return _clean_metric_df(cached, run_name, metric)

    # Ensure API key is set
    if "WANDB_API_KEY" not in os.environ:
        key = _load_api_key_from_cfg()
        if key:
            os.environ["WANDB_API_KEY"] = key

    import wandb  # lazy — heavy import
    api = wandb.Api()
    matched = list(api.runs(f"{entity}/{project}",
                            filters={"display_name": run_name}))
    if not matched:
        raise ValueError(
            f"wandb run '{run_name}' not found in {entity}/{project}")

    wrun = matched[0]
    rows = []
    for row in wrun.scan_history(keys=[metric, "_step"]):
        if metric in row:
            rows.append((row.get("_step", len(rows)), row[metric]))

    result = _clean_metric_df(
        pd.DataFrame(rows, columns=["iteration", "value"]), run_name, metric)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print(f"[rank-analysis] cached -> {cache_path}")
    return result


def merge_rank_perf_df(
    df: pd.DataFrame,
    rank_metric: str,
    perf_dfs: dict[str, pd.DataFrame],
    smooth: int = 5,
) -> pd.DataFrame:
    """Join rank metric from *df* with per-run perf DataFrames.

    Returns DataFrame[run, iteration, rank, reward, progress].
    """
    parts = []
    for run_name, g in df[df["metric"] == rank_metric].groupby("run"):
        if run_name not in perf_dfs:
            raise ValueError(
                f"missing performance data for run '{run_name}' while "
                f"merging '{rank_metric}'")
        rank = (g[["iteration", "value"]]
                .rename(columns={"value": "rank"}))
        rank["iteration"] = pd.to_numeric(rank["iteration"], errors="coerce")
        rank["rank"] = pd.to_numeric(rank["rank"], errors="coerce")
        rank = rank.dropna(subset=["iteration", "rank"]).sort_values(
            "iteration")
        if rank.empty:
            raise ValueError(
                f"empty rank data for run '{run_name}' and metric "
                f"'{rank_metric}'")
        if smooth > 1:
            rank["rank"] = (rank["rank"]
                            .rolling(smooth, min_periods=1, center=True)
                            .mean())
        perf = (perf_dfs[run_name]
                .rename(columns={"value": "reward"}))
        required = {"iteration", "reward"}
        missing = required - set(perf.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(
                f"performance data for run '{run_name}' while merging "
                f"'{rank_metric}' is missing columns: {missing_cols}")
        perf["iteration"] = pd.to_numeric(perf["iteration"], errors="coerce")
        perf["reward"] = pd.to_numeric(perf["reward"], errors="coerce")
        perf = perf.dropna(subset=["iteration", "reward"]).sort_values(
            "iteration")
        if perf.empty:
            raise ValueError(
                f"empty performance data for run '{run_name}' while merging "
                f"'{rank_metric}'")
        merged = pd.merge_asof(rank, perf, on="iteration", direction="nearest")
        merged["run"] = run_name
        merged["progress"] = np.linspace(0, 1, len(merged))
        parts.append(merged)
    if not parts:
        return pd.DataFrame(columns=["run", "iteration", "rank",
                                     "reward", "progress"])
    return pd.concat(parts, ignore_index=True)


def compute_convergence_df(
    merged: pd.DataFrame, convergence_pct: float = 0.15,
) -> pd.DataFrame:
    """Mean ± std of rank and reward over the last *convergence_pct* fraction."""
    rows = []
    for run_name, g in merged.groupby("run"):
        n = max(1, int(len(g) * convergence_pct))
        tail = g.tail(n)
        rows.append({
            "run": run_name,
            "rank_mean": tail["rank"].mean(),
            "rank_std": tail["rank"].std(),
            "reward_mean": tail["reward"].mean(),
            "reward_std": tail["reward"].std(),
        })
    return pd.DataFrame(rows)


def plot_rank_trajectory(merged: pd.DataFrame, rank_metric: str, ax,
                         cmap: str = "viridis"):
    """Plot A: rank (x) vs reward (y) trajectory, colored by training progress."""
    short = rank_metric.rsplit("/", 1)[-1]
    norm = plt.Normalize(0, 1)
    for run_name, g in merged.groupby("run"):
        g = g.sort_values("iteration")
        pts = np.column_stack([g["rank"].values, g["reward"].values])
        if len(pts) < 2:
            continue
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm,
                            linewidth=1.5, alpha=0.8)
        lc.set_array(g["progress"].values[:-1])
        ax.add_collection(lc)
        ax.plot(*pts[0], "o", ms=4, color="k", alpha=0.5)   # start
        ax.plot(*pts[-1], "^", ms=5, color="k", alpha=0.7)  # end
    ax.autoscale()
    ax.set_xlabel(short, fontsize=8)
    ax.set_ylabel("reward", fontsize=8)
    ax.set_title(f"trajectory: {short}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def plot_rank_summary(conv_df: pd.DataFrame, rank_metric: str, ax):
    """Plot B: convergence scatter (mean rank vs mean reward, with errorbars)."""
    short = rank_metric.rsplit("/", 1)[-1]
    for _, row in conv_df.iterrows():
        ax.errorbar(row["rank_mean"], row["reward_mean"],
                    xerr=row["rank_std"], yerr=row["reward_std"],
                    fmt="o", ms=6, capsize=3, alpha=0.8)
        ax.annotate(row["run"], (row["rank_mean"], row["reward_mean"]),
                    fontsize=6, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(short, fontsize=8)
    ax.set_ylabel("reward", fontsize=8)
    ax.set_title(f"convergence: {short}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def plot_rank_timeseries_twin(merged: pd.DataFrame, rank_metric: str, ax):
    """Plot C: twin-axis over time (rank solid, reward dashed)."""
    short = rank_metric.rsplit("/", 1)[-1]
    ax2 = ax.twinx()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (run_name, g) in enumerate(merged.groupby("run")):
        g = g.sort_values("iteration")
        c = colors[i % len(colors)]
        ax.plot(g["iteration"], g["rank"], color=c, linewidth=1.3,
                label=run_name)
        ax2.plot(g["iteration"], g["reward"], color=c, linewidth=1.0,
                 linestyle="--", alpha=0.6)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel(short, fontsize=8)
    ax2.set_ylabel("reward", fontsize=8, alpha=0.6)
    ax.set_title(f"time: {short}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left")


def plot_rank_analysis_grid(
    df: pd.DataFrame,
    rank_metrics: list[str] | None = None,
    perf_metric: str = "Train/mean_reward",
    wandb_project: str = "mymind",
    wandb_entity: str = "tommaselli",
    convergence_pct: float = 0.15,
    smooth: int = 5,
    save_path: str | Path | None = None,
    suptitle: str | None = None,
    refresh_wandb: bool = False,
    log_root: str | Path = DEFAULT_LOG_ROOT,
):
    """Top-level: 3-row × N-col grid of rank × performance analysis."""
    available = set(df["metric"].unique())
    if rank_metrics is None:
        rank_metrics = [m for m in _DEFAULT_RANK_METRICS if m in available]
    else:
        rank_metrics = [m for m in rank_metrics if m in available]
    if not rank_metrics:
        print("[rank-analysis] no rank metrics found in loaded data")
        return

    runs = sorted(df["run"].unique())
    print(f"[rank-analysis] fetching '{perf_metric}' from wandb …")

    perf_dfs: dict[str, pd.DataFrame] = {}
    for run_name in runs:
        try:
            perf_dfs[run_name] = fetch_wandb_perf(
                run_name, perf_metric, wandb_project, wandb_entity,
                cache_dir=Path(log_root) / run_name,
                refresh=refresh_wandb)
        except Exception as e:
            raise RuntimeError(
                f"failed to fetch performance metric '{perf_metric}' for "
                f"run '{run_name}'") from e
    if not perf_dfs:
        print("[rank-analysis] no perf data fetched — nothing to plot")
        return

    ncols = len(rank_metrics)
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.5),
                             squeeze=False)

    for j, rm in enumerate(rank_metrics):
        merged = merge_rank_perf_df(df, rm, perf_dfs, smooth=smooth)
        if merged.empty:
            for i in range(nrows):
                axes[i, j].set_title(
                    f"{rm.rsplit('/', 1)[-1]}\n(no data)", fontsize=8)
                axes[i, j].set_axis_off()
            continue
        conv = compute_convergence_df(merged, convergence_pct)
        plot_rank_trajectory(merged, rm, axes[0, j])
        plot_rank_summary(conv, rm, axes[1, j])
        plot_rank_timeseries_twin(merged, rm, axes[2, j])

    # shared colorbar for row 0 (training progress)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0, :].tolist(), location="right",
                 shrink=0.8, pad=0.02).set_label("training progress",
                                                  fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.95, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[rank-analysis] saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--runs", nargs="*", default=None,
                   help="Run names (folder names under --log-root). "
                        "Omit to use all discovered runs.")
    p.add_argument("--metrics", nargs="*", default=None,
                   help="Metric keys to plot. Omit for a curated default set.")
    p.add_argument("--log-root", type=str, default=str(DEFAULT_LOG_ROOT))
    p.add_argument("--out", type=str, default=None,
                   help="Output image path. Defaults to "
                        "<log-root>/_compare/<runs>.png.")
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--list", action="store_true",
                   help="Just list discovered runs and exit.")
    # rank-analysis flags
    p.add_argument("--rank-analysis", action="store_true",
                   help="Enable rank × performance analysis plots.")
    p.add_argument("--rank-metrics", nargs="*", default=None,
                   help="Rank metric keys. Default: all erank metrics in data.")
    p.add_argument("--perf-metric", type=str, default="Train/mean_reward",
                   help="Wandb performance metric to fetch.")
    p.add_argument("--wandb-project", type=str, default="mymind")
    p.add_argument("--wandb-entity", type=str, default="tommaselli")
    p.add_argument("--convergence-pct", type=float, default=0.15,
                   help="Tail fraction for convergence stats.")
    p.add_argument("--smooth", type=int, default=5,
                   help="Rolling-mean window for rank smoothing.")
    p.add_argument("--refresh-wandb", action="store_true",
                   help="Bypass wandb cache and re-download.")
    args = p.parse_args()

    if args.list:
        for r in list_runs(args.log_root):
            print(r)
        return

    df = load_runs_df(args.runs, log_root=args.log_root)
    if df.empty:
        print("[comparison] no data loaded -- check --log-root / --runs")
        return

    runs = sorted(df["run"].unique())
    print(f"[comparison] runs   : {runs}")
    print(f"[comparison] metrics: {len(df['metric'].unique())}")

    if args.rank_analysis:
        out = args.out or str(
            Path(args.log_root) / "_rank_analysis"
            / ("_vs_".join(runs) + ".png"))
        plot_rank_analysis_grid(
            df,
            rank_metrics=args.rank_metrics,
            perf_metric=args.perf_metric,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            convergence_pct=args.convergence_pct,
            smooth=args.smooth,
            save_path=out,
            suptitle="Rank × Perf: " + " vs ".join(runs),
            refresh_wandb=args.refresh_wandb,
            log_root=args.log_root,
        )
        return

    out = args.out or str(
        Path(args.log_root) / "_compare" / ("_vs_".join(runs) + ".png")
    )
    plot_metrics_grid(
        df, metrics=args.metrics, ncols=args.ncols,
        save_path=out, suptitle=" vs ".join(runs),
    )


if __name__ == "__main__":
    _main()
