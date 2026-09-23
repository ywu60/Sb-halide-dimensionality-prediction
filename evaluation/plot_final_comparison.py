"""The big picture: every ML setting and every LLM condition, one bar each, sorted by test macro-F1.

Writes figures/all_models_comparison.png.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.plot_style import INK, ML_COLOR, LLM_COLOR, new_fig, style_axes, save

SHORT = {"gpt-4.1-2025-04-14": "gpt-4.1", "gpt-5.1-2025-11-13": "gpt-5.1", "gpt-5.5-2026-04-23": "gpt-5.5", "gpt-5.6-sol": "gpt-5.6-sol", "gpt-6-astra": "gpt-6-astra"}
ML_LABEL = {"halide_per_metal": "halide/metal", "inorganic+rdkit_no_pca": "+RDKit no-PCA", "inorganic+rdkit_pca80": "+RDKit PCA80", "inorganic+rdkit_pca90": "+RDKit PCA90", "inorganic+smi_ted_no_pca": "+SMI-TED no-PCA", "inorganic+smi_ted_pca80": "+SMI-TED PCA80", "inorganic+smi_ted_pca90": "+SMI-TED PCA90"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-results", default="results_ml/ml_results.csv")
    ap.add_argument("--llm-results", default="results_llm/llm_results.csv")
    ap.add_argument("--out", default="figures/all_models_comparison.png")
    a = ap.parse_args()

    ml = pd.read_csv(a.ml_results)
    ml_rows = pd.DataFrame({
        "label": [f"{'RF' if m == 'rf' else 'SVM'} | {ML_LABEL.get(s, s)}" for m, s in zip(ml.model, ml.setting)],
        "score": ml.test_macro_f1,
        "family": "ML",
    })

    llm = pd.read_csv(a.llm_results)
    llm_rows = pd.DataFrame({
        "label": [f"{SHORT.get(m, m)} | {rep} | {p}" for m, rep, p in zip(llm.model_resolved, llm.representation, llm.prompting)],
        "score": llm.macro_f1,
        "family": "LLM",
    })

    all_rows = pd.concat([ml_rows, llm_rows], ignore_index=True).sort_values("score", ascending=True).reset_index(drop=True)
    n = len(all_rows)
    fig, ax = new_fig((10.5, max(6, 0.28 * n)))
    colors = [ML_COLOR if f == "ML" else LLM_COLOR for f in all_rows.family]
    ax.barh(range(n), all_rows.score, height=0.68, color=colors, zorder=3)
    for i, v in enumerate(all_rows.score):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(range(n)); ax.set_yticklabels(all_rows.label, fontsize=8)
    ax.set_xlim(0, all_rows.score.max() * 1.12)
    ax.set_xlabel("test macro-F1", fontsize=11.5)
    ax.set_title(f"All {n} model configurations, ML + LLM, on the same held-out test split", fontsize=13.5, loc="left", pad=12)
    import matplotlib.pyplot as plt
    handles = [plt.Rectangle((0, 0), 1, 1, color=ML_COLOR), plt.Rectangle((0, 0), 1, 1, color=LLM_COLOR)]
    ax.legend(handles, ["ML (RandomForest / SVM)", "LLM"], loc="lower right", frameon=False, fontsize=11)
    style_axes(ax, grid_axis="x")
    save(fig, Path(a.out))

    best = all_rows.iloc[-1]
    print(f"Best overall: {best.label} ({best.family}) -- macro-F1 = {best.score:.4f}")


if __name__ == "__main__":
    main()
