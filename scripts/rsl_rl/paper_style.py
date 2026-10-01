"""Paper figure style shared by the Investigator report and comparison plots.

Mirrors the camera-ready figure scripts (`camera-ready/fig{3,5,7}.py`): palette,
serif fonts, sizes, widths, and PNG (300 dpi) + PDF output. W&B native charts are
not affected; only matplotlib figures produced by this repo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "mlp": "#151D3B",
    "simba": "#aa2719",
    "muted_mlp": "#7882A4",
    "muted_simba": "#FF8E9E",
    "ghost": "#909090",
    "text": "#151D3B",
    "edge": "#151D3B",
    "grid": "#DADBBD",
    "background": "#FFFFFF",
}

FULL_WIDTH = 6.8
COLUMN_WIDTH = 3.3

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",  # math follows the (bold) text font
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "text.color": PALETTE["text"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "axes.edgecolor": PALETTE["edge"],
    "axes.linewidth": 0.7,
    "axes.axisbelow": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "legend.frameon": True,
    "legend.edgecolor": PALETTE["edge"],
    "legend.framealpha": 0.9,
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor": PALETTE["background"],
    "savefig.facecolor": PALETTE["background"],
}

# Signed values (Jacobians, activations, weights, signed Gram): navy < 0 < crimson.
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "mtp_diverging",
    [PALETTE["mlp"], PALETTE["muted_mlp"], PALETTE["background"],
     PALETTE["muted_simba"], PALETTE["simba"]],
)
# Non-negative values (Gram similarity, Jacobian std): white -> navy.
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "mtp_sequential", [PALETTE["background"], PALETTE["muted_mlp"], PALETTE["mlp"]]
)

# W&B config ids never appear in figures; legends use the paper names.
PAPER_NAMES = {
    "mlp1": "MLP-XS",
    "mlp2": "MLP-S",
    "mlp_mid": "MLP-M",
    "mlp_deep": "MLP-L",
    "mlp3": "MLP-XXL",
    "pfo_mlp3": "MLP-XXL+PFO",
    "simba1": "SimBa-XS",
    "simba2": "SimBa-S",
    "simba3": "SimBa-M",
    "simba4": "SimBa-L",
    "simba5": "SimBa-XXL",
    "pfo_simba3": "SimBa-M+PFO",
    "linop2": "LinOp-2",
    "linop4": "LinOp-4",
}

_LINESTYLES = ("-", "--", ":", "-.")


def apply_style() -> None:
    plt.rcParams.update(RC)


def paper_name(label: str) -> str:
    return PAPER_NAMES.get(label, label)


def arch_of(label: str) -> str:
    """'SimBa' or 'MLP' from a run label or config id."""
    return "SimBa" if "simba" in label.lower() else "MLP"


def run_styles(labels_and_archs: list[tuple[str, str]]) -> dict[str, dict]:
    """Line style per run: architecture color, line style varies within an architecture."""
    seen: dict[str, int] = {}
    styles = {}
    for label, arch in labels_and_archs:
        k = seen.get(arch, 0)
        seen[arch] = k + 1
        styles[label] = {
            "color": PALETTE["simba" if arch == "SimBa" else "mlp"],
            "linestyle": _LINESTYLES[k % len(_LINESTYLES)],
        }
    return styles


def style_axes(ax, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.tick_params(length=2.0, width=0.6)


FHD_WIDTH_PX = 1920 * 1.1  # margin: bbox_inches="tight" crops the canvas


def figure_dpi(fig) -> float:
    """DPI that gives at least 1920 px width (and never less than 300 dpi)."""
    return max(300.0, FHD_WIDTH_PX / fig.get_figwidth())


def figure_image(fig):
    """Tight-cropped RGB PIL image at figure_dpi, for W&B (which otherwise rasterizes at 100 dpi)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=figure_dpi(fig), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_figure(fig, path: str | Path) -> Path:
    """Write `<path>.png` (>= 1920 px wide) and `<path>.pdf`."""
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=figure_dpi(fig), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


apply_style()
