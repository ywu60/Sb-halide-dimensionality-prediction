"""Reproduce Figure 4 and its SVM supplementary figure: feature importance for the best-performing
ML and LLM configurations.

Figure 4 panels
---------------
(a) Permutation importance for the best ML model, RF (feature_importance/permutation_importance.py's
    figures/ml_permutation_importance_no_title.png).
(b) Grounded LLM feature importance, Borda score by model, for the paired "with labeled examples"
    condition (llm_feature_importance/stage2_plot.py's
    llm_feature_importance/results_with_data/figures/llm_feature_importance_heatmap.png).

Supplementary figure
--------------------
Permutation importance for the best SVM (single test set), drawn in the same style as panel (a)
from results_ml/svm/permutation_importance.csv. If that file does not exist yet, it is computed first
with ``python -m feature_importance.permutation_importance --model svm`` (needs results_ml/ml_results.csv
and the trained models from ml_training.train_models).

Figure 4 itself only composes the two already-rendered panel images; it does not regenerate either
panel. See the README for how to produce them first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

BG = "#ffffff"
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-a",
        type=Path,
        default=REPOSITORY_ROOT / "figures" / "ml_permutation_importance_no_title.png",
        help="ML permutation-importance panel (default: figures/ml_permutation_importance_no_title.png).",
    )
    parser.add_argument(
        "--panel-b",
        type=Path,
        default=REPOSITORY_ROOT / "llm_feature_importance" / "results_with_data" / "figures"
        / "llm_feature_importance_heatmap.png",
        help=(
            "Grounded LLM feature-importance heatmap (default: "
            "llm_feature_importance/results_with_data/figures/llm_feature_importance_heatmap.png)."
        ),
    )
    parser.add_argument(
        "--svm-results",
        type=Path,
        default=REPOSITORY_ROOT / "results_ml" / "svm" / "permutation_importance.csv",
        help="Best-SVM permutation importance (default: results_ml/svm/permutation_importance.csv; computed if missing).",
    )
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "data" / "prepared_data.xlsx",
                        help="Prepared data, used only if the SVM permutation importance has to be computed.")
    parser.add_argument("--ml-results", type=Path, default=REPOSITORY_ROOT / "results_ml" / "ml_results.csv",
                        help="ml_results.csv, used only if the SVM permutation importance has to be computed.")
    parser.add_argument("--top", type=int, default=20, help="Features shown in the SVM figure (same as panel a).")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "output")
    return parser.parse_args()


def make_figure_4(panel_a: Path, panel_b: Path, output: Path) -> None:
    for label, path in (("--panel-a", panel_a), ("--panel-b", panel_b)):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} image: {path}. See the README for how to generate it first.")

    panels = [("a", panel_a), ("b", panel_b)]
    images = [mpimg.imread(path) for _, path in panels]

    # Give each panel a width proportional to its own image's aspect ratio (at a shared row
    # height) so imshow doesn't stretch or squash either one -- the two source panels have
    # quite different native aspect ratios (a tall bar chart vs. a wide heatmap).
    height_in = 6.6
    aspects = [image.shape[1] / image.shape[0] for image in images]
    width_in = height_in * sum(aspects)

    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in), gridspec_kw={"width_ratios": aspects})
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.04)

    for ax, image, (label, _) in zip(axes, images, panels):
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.01, 0.99, f"({label})", transform=ax.transAxes,
            fontsize=20, fontweight="normal", color="black", ha="left", va="top", clip_on=False,
        )

    fig.savefig(output / "figure_4.png", dpi=600, facecolor=BG)
    fig.savefig(output / "figure_4.pdf", facecolor=BG)
    plt.close(fig)
    print(f"Saved Figure 4 to {output.resolve()}")


def make_svm_supplement(svm_results: Path, data: Path, ml_results: Path, top: int, output: Path) -> None:
    if not svm_results.exists():
        print(f"{svm_results} not found; computing best-SVM permutation importance first...")
        if not ml_results.exists():
            raise FileNotFoundError(f"Missing {ml_results}. Run ml_training.train_models first (see the README).")
        subprocess.run(
            [sys.executable, "-m", "feature_importance.permutation_importance",
             "--data", str(data), "--results", str(ml_results), "--model", "svm",
             "--out", str(svm_results.parent), "--figures", str(REPOSITORY_ROOT / "figures" / "svm")],
            cwd=REPOSITORY_ROOT, check=True,
        )

    # Same plotting function (and so the same style) as panel (a), imported from the repository root.
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from feature_importance.permutation_importance import plot_importance

    imp = pd.read_csv(svm_results).sort_values("mean_importance", ascending=False)
    for suffix in ("png", "pdf"):
        plot_importance(imp.head(top), "se_importance", "permutation importance", "",
                        output / f"figure_4_supplement_svm.{suffix}", show_title=False)
    print(f"Saved SVM supplementary figure to {output.resolve()}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    make_figure_4(args.panel_a, args.panel_b, args.output)
    make_svm_supplement(args.svm_results, args.data, args.ml_results, args.top, args.output)


if __name__ == "__main__":
    main()
