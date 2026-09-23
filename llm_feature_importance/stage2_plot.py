"""Create the four Stage 2 feature-importance figures from analyzed CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from llm_feature_importance.stage2_config import (
    DEFAULT_OUTPUT_DIR,
    FEATURE_IDS,
    FIGURE_FEATURE_LABELS,
    MODELS,
    MODEL_NAMES,
)


BG = "#ffffff"
INK = "#111111"
GRID = "#dedede"
BLUE = "#2a78d6"
MODEL_COLORS = {
    "GPT-5.1": "#2a78d6",
    "GPT-5.5": "#e66a2c",
    "GPT-5.6 Sol": "#159d73",
    "GPT-6 Astra": "#6a51a3",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.linewidth": 0.8,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")


def ordered_importance(df: pd.DataFrame) -> pd.DataFrame:
    expected = {(model["model_id"], feature_id) for model in MODELS for feature_id in FEATURE_IDS}
    actual = set(zip(df.model_id, df.feature_id))
    if actual != expected or len(df) != 28:
        raise RuntimeError("model_feature_importance.csv does not contain exactly the expected 28 rows")
    return df.copy()


def plot_importance_heatmap(df: pd.DataFrame, output: Path) -> None:
    pivot = (
        df.pivot(index="feature_id", columns="model", values="mean_borda")
        .reindex(index=FEATURE_IDS, columns=MODEL_NAMES)
    )
    labels = [FIGURE_FEATURE_LABELS[feature_id] for feature_id in FEATURE_IDS]
    model_labels = [
        name.replace("GPT-5.6 Sol", "GPT-5.6\nSol").replace("GPT-6 Astra", "GPT-6\nAstra")
        for name in MODEL_NAMES
    ]
    cmap = LinearSegmentedColormap.from_list("white_blue", ["#f7fbff", BLUE])

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    image = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(FEATURE_IDS)):
        for j in range(len(MODEL_NAMES)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=14,
                    color=BG if value >= 0.62 else INK)
    ax.set_xticks(range(len(MODEL_NAMES)), model_labels, fontsize=13)
    ax.set_yticks(range(len(FEATURE_IDS)), labels, fontsize=18)
    ax.set_xticks(np.arange(-0.5, len(MODEL_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FEATURE_IDS), 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    colorbar.set_label("Mean normalized Borda score", fontsize=17)
    colorbar.ax.tick_params(labelsize=15)
    save(fig, output / "figure1_feature_importance_heatmap")


def plot_importance_ci(df: pd.DataFrame, output: Path) -> None:
    labels = [FIGURE_FEATURE_LABELS[feature_id] for feature_id in FEATURE_IDS]
    y = np.arange(len(FEATURE_IDS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    for offset, model in zip(offsets, MODELS):
        group = df[df.model_id.eq(model["model_id"])].set_index("feature_id").reindex(FEATURE_IDS)
        means = group.mean_borda.to_numpy(dtype=float)
        lower = group.ci_lower.to_numpy(dtype=float)
        upper = group.ci_upper.to_numpy(dtype=float)
        ax.errorbar(
            means,
            y + offset,
            xerr=np.vstack([means - lower, upper - means]),
            fmt="o",
            markersize=6,
            capsize=2.5,
            elinewidth=1.4,
            color=MODEL_COLORS[model["model"]],
            label=model["model"],
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(y, labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalized Borda score (95% bootstrap CI)", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=10)
    ax.set_title("Model-specific feature importance and uncertainty", loc="left", fontsize=14, pad=12)
    save(fig, output / "figure2_model_feature_importance_ci")


def plot_stability(df: pd.DataFrame, output: Path) -> None:
    ordered = df.set_index("model").reindex(MODEL_NAMES)
    means = ordered.mean_kendall_tau.to_numpy(dtype=float)
    lower = ordered.ci_lower.to_numpy(dtype=float)
    upper = ordered.ci_upper.to_numpy(dtype=float)
    x = np.arange(len(MODEL_NAMES))
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, model_name in enumerate(MODEL_NAMES):
        color = MODEL_COLORS[model_name]

        ax.vlines(x[i], lower[i], upper[i], color=color, linewidth=1.6, zorder=2)
        ax.hlines([lower[i], upper[i]], x[i] - 0.06, x[i] + 0.06, color=color, linewidth=1.6, zorder=2)
        ax.scatter(x[i], means[i], s=58, color=color, zorder=3)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(x, MODEL_NAMES, fontsize=10.5)
    ax.set_ylim(-1, 1.03)
    ax.set_ylabel("Mean pairwise Kendall's τ (95% bootstrap CI)", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Within-model ranking stability", loc="left", fontsize=14, pad=12)
    save(fig, output / "figure3_model_stability_kendall")


def plot_intermodel_kendall(matrix: pd.DataFrame, output: Path) -> None:
    matrix = matrix.set_index("model").reindex(index=MODEL_NAMES, columns=MODEL_NAMES)
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.1, 6.1))
    image = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1, aspect="equal")
    for i in range(len(MODEL_NAMES)):
        for j in range(len(MODEL_NAMES)):
            value = values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=12,
                    color=BG if abs(value) >= 0.62 else INK)
    ax.set_xticks(range(len(MODEL_NAMES)), MODEL_NAMES, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(MODEL_NAMES)), MODEL_NAMES, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(MODEL_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_NAMES), 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Kendall's τ", fontsize=10.5)
    ax.set_title("Inter-model agreement in final feature rankings", loc="left", fontsize=13, pad=12)
    save(fig, output / "figureS1_intermodel_kendall_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    figures = args.figures or args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    importance = ordered_importance(pd.read_csv(args.out / "model_feature_importance.csv"))
    stability = pd.read_csv(args.out / "model_stability_kendall.csv")
    intermodel = pd.read_csv(args.out / "intermodel_kendall.csv")
    plot_importance_heatmap(importance, figures)
    plot_importance_ci(importance, figures)
    plot_stability(stability, figures)
    plot_intermodel_kendall(intermodel, figures)


if __name__ == "__main__":
    main()
