"""Confusion matrices for the single best ML model and the single best LLM condition (by test macro-F1).
Writes figures/confusion_matrix_best_ml.png and figures/confusion_matrix_best_llm.png.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from common.plot_style import BG, INK, new_fig, save

LABELS = ["0D", "non-0D"]  # target_non0D: 0 = 0D, 1 = non-0D
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]  # light -> dark


def plot_cm(cm: np.ndarray, title: str, subtitle: str, out_path: Path):
    fig, ax = new_fig((6.4, 5.8))
    vmax = cm.max()
    # Sequential single-hue heatmap (magnitude), not a rainbow -- consistent with the palette's sequential rule.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax * 1.05)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum() if cm[i].sum() else 0
            txt_color = "white" if cm[i, j] / vmax > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]}\n({share * 100:.0f}%)", ha="center", va="center", fontsize=14, color=txt_color, linespacing=1.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS, fontsize=11.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABELS, fontsize=11.5)
    ax.set_xlabel("predicted", fontsize=11.5); ax.set_ylabel("true", fontsize=11.5)
    ax.set_title(f"{title}\n{subtitle}", fontsize=12.5, loc="left", pad=12)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True); ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.tick_params(which="minor", length=0)
    save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-results", default="results_ml/ml_results.csv")
    ap.add_argument("--ml-predictions", default="results_ml/ml_test_predictions.csv")
    ap.add_argument("--llm-results", default="results_llm/llm_results.csv")
    ap.add_argument("--llm-predictions", default="results_llm/llm_test_predictions.csv")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    figures = Path(a.figures); figures.mkdir(parents=True, exist_ok=True)

    ml_res = pd.read_csv(a.ml_results)
    best_ml = ml_res.sort_values("test_macro_f1", ascending=False).iloc[0]
    ml_pred = pd.read_csv(a.ml_predictions)
    sub = ml_pred[(ml_pred.model == best_ml.model) & (ml_pred.setting == best_ml.setting)]
    cm = confusion_matrix(sub.true, sub.prediction, labels=[0, 1])
    plot_cm(cm, f"Best ML model: {best_ml.model} / {best_ml.setting}", f"test macro-F1 = {best_ml.test_macro_f1:.3f}, n = {len(sub)}", figures / "confusion_matrix_best_ml.png")

    llm_res = pd.read_csv(a.llm_results)
    best_llm = llm_res.sort_values("macro_f1", ascending=False).iloc[0]
    llm_pred = pd.read_csv(a.llm_predictions)
    llm_pred = llm_pred[llm_pred.status.eq("ok")]
    sub = llm_pred[(llm_pred.model_resolved == best_llm.model_resolved) & (llm_pred.representation == best_llm.representation) & (llm_pred.prompting == best_llm.prompting)]
    cm = confusion_matrix(sub.true, sub.prediction, labels=[0, 1])
    plot_cm(cm, f"Best LLM condition: {best_llm.model_resolved}", f"{best_llm.representation}, {best_llm.prompting} -- macro-F1 = {best_llm.macro_f1:.3f}, n = {len(sub)}", figures / "confusion_matrix_best_llm.png")

    print(f"Best ML: {best_ml.model}/{best_ml.setting} (f1={best_ml.test_macro_f1:.3f})")
    print(f"Best LLM: {best_llm.model_resolved}/{best_llm.representation}/{best_llm.prompting} (f1={best_llm.macro_f1:.3f})")


if __name__ == "__main__":
    main()
