"""Bar charts for the LLM results in results_llm/llm_results.csv.

Two figures:
  llm_results_comparison.png   grouped bars -- macro-F1 for each model x representation, one bar per
                                prompting condition (zero / 12-shot / all-shot), with the best ML model
                                drawn as a reference line.
  llm_allshot_gain.png         diverging bars -- (all-shot minus 12-shot) per condition.

    python 08_plot_llm_results.py --llm-results results_llm/llm_results.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common.plot_style import INK, INK_SOFT, PROMPTING_RAMP, POSITIVE, NEGATIVE, new_fig, style_axes, save

SHORT = {"gpt-4.1-2025-04-14": "gpt-4.1", "gpt-5.1-2025-11-13": "gpt-5.1", "gpt-5.5-2026-04-23": "gpt-5.5", "gpt-5.6-sol": "gpt-5.6-sol", "gpt-6-astra": "gpt-6-astra"}
PROMPTING = [("zero", "zero-shot", PROMPTING_RAMP["zero"]), ("12shot", "12-shot", PROMPTING_RAMP["few_shot"]), ("allshot", "all-shot", PROMPTING_RAMP["all_shot"])]

# Three-stop, light-to-dark sequential ramps used for publication-oriented alternatives.
# They follow the ordered-palette principle recommended by Nature and ColorBrewer.
SEQUENTIAL_PALETTES = {
    # Selected final palette: neutral baseline followed by the original blue progression.
    "selected_gray_blue_navy": ["#d9dde2", "#2a78d6", "#104281"],
    "blue": ["#c6dbef", "#4292c6", "#084594"],
    "purple_magenta": ["#e7e1ef", "#c994c7", "#980043"],
    "yellow_orange_brown": ["#fee8c8", "#fdbb84", "#d94701"],
    # Fixed-hue ramps: the three conditions differ by lightness, not by hue.
    "violet_single_hue": ["#eedff6", "#a859cf", "#54206f"],
    "teal_single_hue": ["#dff6f4", "#59cfc5", "#206f68"],
    "violet_lighter_middle": ["#eedff6", "#c58ade", "#54206f"],
    # Retains the original neutral 0-shot and navy all-shot bars; only 12-shot is lightened.
    "blue_lighter_middle": ["#d9dde2", "#78aee4", "#104281"],
}


def load(path: str):
    return load_dataframe(pd.read_csv(path))


def load_dataframe(df: pd.DataFrame):
    df = df.copy()
    df["model"] = df.model_resolved.map(lambda m: SHORT.get(m, m))
    order = [m for m in SHORT.values() if m in set(df.model)]
    df["model"] = pd.Categorical(df.model, order, ordered=True)
    df["representation"] = pd.Categorical(df.representation, ["name", "smiles"], ordered=True)
    return df.sort_values(["model", "representation"])


def group_labels(pairs):
    return [f"{m}\n{'IUPAC name' if r == 'name' else 'SMILES'}" for m, r in pairs]


def plot_grouped(df: pd.DataFrame, ml_baseline: float, ml_label: str, out: Path):
    score = df.set_index(["model", "representation", "prompting"]).macro_f1
    pairs = list(dict.fromkeys(zip(df.model, df.representation)))
    x = np.arange(len(pairs)); w = 0.26  # widened from 0.20 -- that left too big a gap (0.40)
    # before the next group; this leaves a smaller but still visible one (~0.22).

    fig, ax = new_fig((max(14.5, 1.25 * len(pairs)), 7.4))
    bars_by_cond, vals_by_cond = [], []
    for i, (key, label, color) in enumerate(PROMPTING):
        vals = [score.get((m, r, key), np.nan) for m, r in pairs]
        offset = (i - 1) * w  # no gap between the 3 bars within a group -- edge to edge
        bars = ax.bar(x + offset, vals, w, color=color, label=label, zorder=3)
        bars_by_cond.append(bars); vals_by_cond.append(vals)

    # Labels are placed after all bars are drawn (rather than per-condition), group by group,
    # because avoiding overlap needs to reason about all 3 bars in the group together. At this
    # font size a "0.xxx" label is wider than the (now-narrow, flush) bars, so a dead-center
    # label always overhangs into its neighbor(s) -- the fix is directional, not just "left":
    # the left bar's label is pushed further left (into this group's own share of the inter-group
    # gap), the right bar's label pushed further right (same, on the other side), and the middle
    # bar's label stays centered -- which also moves the outer two labels' overhang away from the
    # middle one instead of into it.
    OUTWARD_SHIFT = [-0.04, 0.0, 0.04]  # smaller than before -- wider bars overhang less
    COLLIDE_GAP = 0.012  # value gap below which same-group neighbors visually collide at this font size
    for j in range(len(pairs)):
        group_vals = [vals_by_cond[i][j] for i in range(3)]
        group_bars = [bars_by_cond[i][j] for i in range(3)]
        extra = [0.0, 0.0, 0.0]
        for k in (0, 1):
            va, vb = group_vals[k], group_vals[k + 1]
            if np.isfinite(va) and np.isfinite(vb) and abs(va - vb) < COLLIDE_GAP:
                extra[k + 1] += 0.02  # stagger the right-hand one up, clear of its left neighbor
        for k, (b, v, e) in enumerate(zip(group_bars, group_vals, extra)):
            if np.isfinite(v): ax.text(b.get_x() + b.get_width() / 2 + OUTWARD_SHIFT[k], v + 0.008 + e, f"{v:.3f}", ha="center", va="bottom", fontsize=12, color=INK)

    ax.axhline(ml_baseline, color=INK_SOFT, lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.text(1.008, ml_baseline, ml_label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=14, color=INK_SOFT, style="italic", linespacing=1.4)

    ax.set_ylim(0, max(0.9, score.max() * 1.12))
    ax.set_ylabel("test macro-F1", fontsize=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xticks(x); ax.set_xticklabels(group_labels(pairs), fontsize=15)
    # at this font size the legend box is wide/tall enough to sit on top of the tallest
    # bar's value label (gpt-5.1 IUPAC name, all-shot = 0.802) when anchored inside the
    # axes -- park it just above the top spine instead, fully clear of every bar.
    ax.legend(loc="lower left", frameon=False, fontsize=15, ncol=3, title="prompting condition", title_fontsize=18, bbox_to_anchor=(0, 1.01))
    style_axes(ax)
    fig.subplots_adjust(right=0.925)
    save(fig, out)


def plot_gain(df: pd.DataFrame, out: Path):
    score = df.set_index(["model", "representation", "prompting"]).macro_f1
    pairs = list(dict.fromkeys(zip(df.model, df.representation)))
    gains = [score.get((m, r, "allshot"), np.nan) - score.get((m, r, "12shot"), np.nan) for m, r in pairs]
    if not any(np.isfinite(gains)): return
    x = np.arange(len(pairs))

    fig, ax = new_fig((13.5, 5.4))
    bars = ax.bar(x, gains, 0.52, color=[POSITIVE if g >= 0 else NEGATIVE for g in gains], zorder=3)
    for b, g in zip(bars, gains):
        if np.isfinite(g):
            va, pad = ("bottom", 0.004) if g >= 0 else ("top", -0.004)
            ax.text(b.get_x() + b.get_width() / 2, g + pad, f"{g:+.3f}", ha="center", va=va, fontsize=10, color=INK)

    ax.axhline(0, color=INK_SOFT, lw=1.2, zorder=4)
    finite = [g for g in gains if np.isfinite(g)]
    lo, hi = min(finite), max(finite)
    ax.set_ylim(lo - 0.045, hi + 0.045)
    ax.set_ylabel("macro-F1:  all-shot minus 12-shot", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(group_labels(pairs), fontsize=10)
    ax.set_title("Where all-shot prompting pays off relative to 12 curated shots", fontsize=14, pad=20, loc="left")
    style_axes(ax)
    save(fig, out)


def plot_by_representation(df: pd.DataFrame, ml_baseline: float, out: Path, colors=None):
    """A less dense alternative to the all-in-one grouped-bar chart."""
    score = df.set_index(["model", "representation", "prompting"]).macro_f1
    models = list(df.model.cat.categories)
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.2), sharey=True)
    fig.patch.set_facecolor("#ffffff")
    # Touching bars make each model read as one compact three-condition group.
    x = np.arange(len(models)) * 0.96; width = 0.22
    colors = SEQUENTIAL_PALETTES["blue"] if colors is None else colors
    refined_prompting = [(key, label, color) for (key, label, _), color in zip(PROMPTING, colors)]

    for ax, representation, title in zip(axes, ["name", "smiles"], ["IUPAC name input", "SMILES input"]):
        ax.set_facecolor("#ffffff")
        for i, (key, label, color) in enumerate(refined_prompting):
            vals = [score.get((model, representation, key), np.nan) for model in models]
            bars = ax.bar(x + (i - 1) * width, vals, width, color=color, label=label,
                          linewidth=0, zorder=3)
            for j, (bar, value) in enumerate(zip(bars, vals)):
                # All-shot is the headline condition.  Showing only these ten values retains
                # the exact takeaway while removing the text wall above every group.
                if key == "allshot" and np.isfinite(value):
                    # Place labels just right of the narrow navy bar, clear of the 12-shot bar.
                    ax.text(bar.get_x() + bar.get_width() / 2 + 0.075, value + 0.011, f"{value:.3f}",
                            ha="center", va="bottom", fontsize=12, color=INK)

        ax.axhline(ml_baseline, color=INK_SOFT, lw=1.3, ls=(0, (5, 3)), zorder=2)
        ax.set_title(title, fontsize=18, pad=17, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=14)
        ax.set_ylim(0, 0.86)
        ax.tick_params(axis="y", labelsize=14)
        style_axes(ax)

    axes[0].set_ylabel("test macro-F1", fontsize=17)
    axes[1].text(1.015, ml_baseline, f"best ML\n{ml_baseline:.3f}",
                 transform=axes[1].get_yaxis_transform(), ha="left", va="center",
                 fontsize=13, color=INK_SOFT, style="italic")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               title="prompting condition", title_fontsize=17, fontsize=15,
               bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.79, bottom=0.14, wspace=0.10)
    # Unlike the single-axis charts, this figure reserves explicit space for a figure-level
    # legend; calling tight_layout afterwards would discard that reservation.
    fig.savefig(out, dpi=200, facecolor="#ffffff", bbox_inches="tight")
    print(f"Saved {out}")


def plot_dot_by_representation(df: pd.DataFrame, ml_baseline: float, out: Path, colors):
    """Zoomed alternative: position (rather than bar height) encodes performance."""
    score = df.set_index(["model", "representation", "prompting"]).macro_f1
    models = list(df.model.cat.categories)
    x = np.arange(len(models))
    conditions = [(key, label, color) for (key, label, _), color in zip(PROMPTING, colors)]
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.2), sharey=True)
    fig.patch.set_facecolor("#ffffff")

    for ax, representation, title in zip(axes, ["name", "smiles"], ["IUPAC name input", "SMILES input"]):
        ax.set_facecolor("#ffffff")
        for offset, (key, label, color) in zip([-0.16, 0, 0.16], conditions):
            vals = [score.get((model, representation, key), np.nan) for model in models]
            ax.scatter(x + offset, vals, s=100, color=color, edgecolor="white", linewidth=1.0,
                       label=label, zorder=3)
            if key == "allshot":
                for xi, value in zip(x + offset, vals):
                    ax.text(xi + 0.04, value + 0.010, f"{value:.3f}", fontsize=11,
                            ha="left", va="bottom", color=INK)
        ax.axhline(ml_baseline, color=INK_SOFT, lw=1.3, ls=(0, (5, 3)), zorder=2)
        ax.set_title(title, fontsize=18, pad=17, loc="left")
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=14)
        ax.set_ylim(0.30, 0.85); ax.tick_params(axis="y", labelsize=14)
        style_axes(ax)

    axes[0].set_ylabel("test macro-F1", fontsize=17)
    axes[1].text(1.015, ml_baseline, f"best ML\n{ml_baseline:.3f}", transform=axes[1].get_yaxis_transform(),
                 ha="left", va="center", fontsize=13, color=INK_SOFT, style="italic")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, title="prompting condition",
               title_fontsize=17, fontsize=15, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.012, "Zoomed y-axis: 0.30–0.85", ha="center", fontsize=11, color=INK_SOFT)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.79, bottom=0.14, wspace=0.10)
    fig.savefig(out, dpi=200, facecolor="#ffffff", bbox_inches="tight")
    print(f"Saved {out}")


def plot_line_by_representation(df: pd.DataFrame, ml_baseline: float, out: Path, colors):
    """Zoomed alternative emphasizing how prompting changes each model's score."""
    score = df.set_index(["model", "representation", "prompting"]).macro_f1
    models = list(df.model.cat.categories)
    x = np.arange(len(models))
    conditions = [(key, label, color) for (key, label, _), color in zip(PROMPTING, colors)]
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.2), sharey=True)
    fig.patch.set_facecolor("#ffffff")

    for ax, representation, title in zip(axes, ["name", "smiles"], ["IUPAC name input", "SMILES input"]):
        ax.set_facecolor("#ffffff")
        for key, label, color in conditions:
            vals = [score.get((model, representation, key), np.nan) for model in models]
            ax.plot(x, vals, color=color, lw=2.6, marker="o", markersize=8.5,
                    markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3)
        ax.axhline(ml_baseline, color=INK_SOFT, lw=1.3, ls=(0, (5, 3)), zorder=2)
        ax.set_title(title, fontsize=18, pad=17, loc="left")
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=14)
        ax.set_ylim(0.30, 0.85); ax.tick_params(axis="y", labelsize=14)
        style_axes(ax)

    axes[0].set_ylabel("test macro-F1", fontsize=17)
    axes[1].text(1.015, ml_baseline, f"best ML\n{ml_baseline:.3f}", transform=axes[1].get_yaxis_transform(),
                 ha="left", va="center", fontsize=13, color=INK_SOFT, style="italic")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, title="prompting condition",
               title_fontsize=17, fontsize=15, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.012, "Zoomed y-axis: 0.30–0.85", ha="center", fontsize=11, color=INK_SOFT)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.79, bottom=0.14, wspace=0.10)
    fig.savefig(out, dpi=200, facecolor="#ffffff", bbox_inches="tight")
    print(f"Saved {out}")


def plot_prompting_heatmap(df: pd.DataFrame, ml_baseline: float, out: Path):
    """A compact lookup figure for all model/representation/prompting scores."""
    score = df.set_index(["model", "representation", "prompting"]).macro_f1
    rows = [(model, representation) for model in df.model.cat.categories for representation in ["name", "smiles"]]
    cols = [key for key, _, _ in PROMPTING]
    values = np.array([[score.get((model, representation, key), np.nan) for key in cols]
                       for model, representation in rows])
    labels = [label for _, label, _ in PROMPTING]

    fig, ax = new_fig((7.4, 8.0))
    image = ax.imshow(values, cmap="Blues", vmin=0.35, vmax=0.82, aspect="auto")
    for row, (model, representation) in enumerate(rows):
        for col in range(len(cols)):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            text_color = "white" if value >= 0.69 else INK
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=12, color=text_color)
            if value >= ml_baseline:
                ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="#e69f00", lw=2.2))

    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, fontsize=13)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{model}  —  {'IUPAC name' if rep == 'name' else 'SMILES'}" for model, rep in rows], fontsize=11)
    ax.set_title("Prompting comparison (test macro-F1)", fontsize=15, pad=28, loc="left")
    ax.set_xlabel("Orange outline: exceeds best ML macro-F1 (0.683)", fontsize=11, labelpad=13)
    ax.xaxis.set_label_position("bottom")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, shrink=0.9)
    colorbar.set_label("test macro-F1", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-results", default="results_llm/llm_results.csv")
    ap.add_argument("--gpt6-astra-results", default="results_llm/gpt6_astra_prompting_summary.csv", help="Optional Astra macro-F1 summary to append to the baseline results")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    figures = Path(a.figures); figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.llm_results)
    astra_path = Path(a.gpt6_astra_results)
    if astra_path.exists():
        astra = pd.read_csv(astra_path)
        # The historical baseline contains accuracy, whereas the Astra comparison artifact
        # intentionally records only the requested macro-F1 metric. Align columns before concat.
        df = pd.concat([df, astra.reindex(columns=df.columns)], ignore_index=True)
    df = load_dataframe(df)
    incomplete = df[df.n != df.n.max()]
    if len(incomplete):
        print(f"WARNING: {len(incomplete)} condition(s) have fewer than {df.n.max()} predictions and are not comparable:")
        print(incomplete[["model", "representation", "prompting", "n"]].to_string(index=False))

    ml_baseline = 0.683
    ml_label = f"best ML\n{ml_baseline:.3f}"

    plot_grouped(df, ml_baseline, ml_label, figures / "llm_results_comparison.png")
    plot_gain(df, figures / "llm_allshot_gain.png")
    plot_by_representation(
        df, ml_baseline, figures / "llm_results_by_representation_final.png",
        SEQUENTIAL_PALETTES["selected_gray_blue_navy"],
    )
    plot_dot_by_representation(df, ml_baseline, figures / "llm_results_dot_zoomed.png", SEQUENTIAL_PALETTES["selected_gray_blue_navy"])
    plot_line_by_representation(df, ml_baseline, figures / "llm_results_line_zoomed.png", SEQUENTIAL_PALETTES["selected_gray_blue_navy"])
    plot_prompting_heatmap(df, ml_baseline, figures / "llm_prompting_heatmap.png")


if __name__ == "__main__":
    main()
