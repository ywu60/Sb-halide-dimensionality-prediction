"""SHAP feature importance for the best ML model (ranked by test_macro_f1 in ml_results.csv).

Complements the permutation importance in 04_permutation_importance.py: SHAP attributes each individual
test-compound prediction additively (so it also shows *direction* -- does a high X:M ratio push toward 0D or
away?), computed on held-out test compounds. TreeExplainer is exact for RandomForest; for a linear-kernel SVM
LinearExplainer is exact; for anything else (e.g. an RBF-kernel SVM, which has no closed-form attribution) a
model-agnostic permutation explainer is used as a fallback -- slower, but correct for any scikit-learn model.

The attributed score is P(non-0D) for RandomForest (TreeExplainer explains predict_proba) and the raw SVM
decision-function margin for SVM (no calibrated predict_proba is fit in 03_run_ml.py); both are monotonic in
"how confidently non-0D", so direction (sign) is comparable even though the two are not on the same scale.

    python 11_shap_importance.py --data prepared_data.xlsx --results results_ml/ml_results.csv

Writes:
  results_ml/shap_importance.csv
  figures/ml_shap_importance.png    bar chart, mean |SHAP value| per feature, colored by feature block
  figures/ml_shap_beeswarm.png      per-compound attributions, showing direction as well as magnitude
"""
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from common.plot_style import BG, INK, INORGANIC_COLOR, CATION_COLOR, new_fig, save, style_axes
from feature_importance import permutation_importance as perm_mod


def shap_values_for_non0d(pipe, X_background: np.ndarray, X_explain: np.ndarray, names: list[str]):
    """Return a (n_explain, n_features) attribution array, oriented toward the non-0D (label 1) class."""
    model = pipe.named_steps["model"]
    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_explain, check_additivity=False)
        index = int(np.flatnonzero(model.classes_ == 1)[0])
        values = np.asarray(values)
        return (values[index] if isinstance(values, list) else values[:, :, index] if values.ndim == 3 else values), "P(non-0D)"
    if hasattr(model, "coef_"):  # linear-kernel SVC: exact linear attribution
        explainer = shap.LinearExplainer(model, X_background)
        return np.asarray(explainer.shap_values(X_explain)), "SVM decision-function margin"
    # RBF (or any other) kernel: no closed form, fall back to the model-agnostic permutation explainer.
    background = shap.sample(X_background, min(50, len(X_background)), random_state=42)
    explainer = shap.Explainer(model.decision_function, background, algorithm="permutation", feature_names=names)
    return np.asarray(explainer(X_explain, max_evals=2 * len(names) + 1).values), "SVM decision-function margin"


def plot_bar(imp: pd.DataFrame, out_path: Path):
    d = imp.sort_values("mean_abs_shap")
    colors = [CATION_COLOR if b == "cation" else INORGANIC_COLOR for b in d.block]
    fig, ax = new_fig((9.4, max(4.5, 0.42 * len(d) + 1.5)))
    ax.barh(range(len(d)), d.mean_abs_shap, height=0.68, color=colors, zorder=3)
    span = d.mean_abs_shap.max()
    for i, v in enumerate(d.mean_abs_shap):
        ax.text(v + span * 0.012, i, f"{v:.4f}", va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.feature, fontsize=9.5)
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel("mean |SHAP value|  (test split)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=INORGANIC_COLOR), plt.Rectangle((0, 0), 1, 1, color=CATION_COLOR)]
    ax.legend(handles, ["Inorganic / water feature", "Cation representation (RDKit / SMI-TED)"], loc="lower right", frameon=False, fontsize=10)
    style_axes(ax, grid_axis="x")
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prepared_data.xlsx")
    ap.add_argument("--results", default="results_ml/ml_results.csv")
    ap.add_argument("--out-csv", default="results_ml/shap_importance.csv")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()

    results = pd.read_csv(a.results)
    best = results.sort_values("test_macro_f1", ascending=False).iloc[0]
    print(f"Best ML model: {best.model} / {best.setting}  (test_macro_f1={best.test_macro_f1:.4f})")

    bundle = joblib.load(best.saved_model)
    pipe, columns = bundle["model"], bundle["input_columns"]

    df = pd.read_excel(a.data)
    train = df[df.split.eq("train")].reset_index(drop=True)
    test = df[df.split.eq("test")].reset_index(drop=True)
    X_train, names_tr = perm_mod.transformed_matrix_and_names(pipe, train, columns)
    X_test, names = perm_mod.transformed_matrix_and_names(pipe, test, columns)
    assert names_tr == names

    values, target_label = shap_values_for_non0d(pipe, X_train, X_test, names)
    print(f"Explained score: {target_label}")

    imp = pd.DataFrame({
        "feature": names,
        "block": ["cation" if n.startswith(("smi_ted", "rdkit")) else "inorganic" for n in names],
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    out_csv = Path(a.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_csv, index=False)
    print(imp.round(5).to_string(index=False))
    grouped = imp.groupby("block").mean_abs_shap.agg(["sum", "mean", "size"])
    print("\nBy block (sum / per-feature mean / count):")
    print(grouped.round(4).to_string())

    figures = Path(a.figures); figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plot_bar(imp, figures / "ml_shap_importance.png")
    ax.set_title(f"SHAP feature importance ({target_label})\nbest ML model ({best.model}, {best.setting})", fontsize=13, loc="left", pad=12, linespacing=1.4)
    save(fig, figures / "ml_shap_importance.png")

    fig = plt.figure(figsize=(9.6, max(4.5, 0.42 * len(names) + 1.5)))
    shap.summary_plot(values, features=X_test, feature_names=names, show=False, plot_size=None, max_display=len(names))
    fig.patch.set_facecolor(BG)
    for ax in fig.axes: ax.set_facecolor(BG)
    fig.suptitle(f"SHAP attributions ({target_label}) -- best ML model ({best.model}, {best.setting})", fontsize=12.5, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(figures / "ml_shap_beeswarm.png", dpi=200, facecolor=BG, bbox_inches="tight")
    print(f"\nSaved {out_csv}, figures/ml_shap_importance.png, figures/ml_shap_beeswarm.png")


if __name__ == "__main__":
    main()
