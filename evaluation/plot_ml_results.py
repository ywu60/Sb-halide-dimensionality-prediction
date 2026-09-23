"""Bar chart comparing RandomForest vs SVM test macro-F1 across every ML feature setting.

    python 07_plot_ml_results.py --results results_ml/ml_results.csv --out figures/ml_results_comparison.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.plot_style import INK, BLUE, ORANGE, new_fig, style_axes, save

SETTINGS = [
    ("halide_per_metal", "halide/metal\n(baseline)"),
    ("inorganic+rdkit_no_pca", "+RDKit\nno PCA"),
    ("inorganic+rdkit_pca80", "+RDKit\nPCA-80"),
    ("inorganic+rdkit_pca90", "+RDKit\nPCA-90"),
    ("inorganic+smi_ted_no_pca", "+SMI-TED\nno PCA"),
    ("inorganic+smi_ted_pca80", "+SMI-TED\nPCA-80"),
    ("inorganic+smi_ted_pca90", "+SMI-TED\nPCA-90"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_ml/ml_results.csv")
    ap.add_argument("--out", default="figures/ml_results_comparison.png")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    score = df.set_index(["model", "setting"]).test_macro_f1

    order = [s for s, _ in SETTINGS]; labels = [lab for _, lab in SETTINGS]
    rf = [score.get(("rf", s), np.nan) for s in order]
    svm = [score.get(("svm", s), np.nan) for s in order]

    x = np.arange(len(order)); w = 0.4
    fig, ax = new_fig((12, 6.2))
    for offset, vals, color, name in [(-w / 2, rf, BLUE, "Random Forest"), (w / 2, svm, ORANGE, "SVM (threshold-tuned)")]:
        bars = ax.bar(x + offset, vals, w, color=color, label=name, zorder=3)
        for b, v in zip(bars, vals):
            if np.isfinite(v): ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=INK)

    ax.set_ylim(0, max(max(rf), max(svm)) * 1.22)
    ax.set_ylabel("test macro-F1", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=11, ncol=2)
    ax.set_title("ML models: test macro-F1 across feature settings (inorganic + water baseline, then + cation representation)", fontsize=12.5, loc="left", pad=12)
    style_axes(ax)
    save(fig, Path(a.out))


if __name__ == "__main__":
    main()
