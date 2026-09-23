"""Permutation feature importance for the best ML model (ranked by test_macro_f1 in ml_results.csv).

--data prepared_data.xlsx --results results_ml/ml_results.csv --out results_ml

Writes:
  results_ml/permutation_importance.csv                single test-set estimate
  results_ml/permutation_importance_cv.csv             cross-validated estimate
  figures/ml_permutation_importance.png                single test-set estimate, top-20, +-1 SE error bars
  figures/ml_permutation_importance_cv.png             cross-validated estimate, +-1 between-fold SE error bars
"""
import argparse, json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from common.plot_style import INK, INORGANIC_COLOR, CATION_COLOR, new_fig, save, style_axes
from ml_training import train_models as ml

FEATURE_LABEL = {"inorganic_halide_per_metal": "Halide:Sb ratio"}


def transformed_matrix_and_names(pipe: Pipeline, df: pd.DataFrame, columns: list[str]):

    ct = pipe.named_steps["features"]
    X_raw = df[columns].to_numpy(float)
    X = ct.transform(X_raw)
    n_common = len(ct.transformers_[0][2])
    rep_step = ct.named_transformers_.get("representation")
    pca = rep_step.named_steps.get("pca") if hasattr(rep_step, "named_steps") else None
    if pca is not None:
        rep_cols = columns[n_common:]
        prefix = "rdkit" if any(c.startswith("rdkit_") for c in rep_cols) else "smi_ted" if any(c.startswith("smi_ted_") for c in rep_cols) else "rep"
        names = columns[:n_common] + [f"{prefix}_PC{i + 1}" for i in range(pca.n_components_)]
    else:
        names = columns
    if X.shape[1] != len(names): raise RuntimeError(f"Transformed width {X.shape[1]} does not match {len(names)} feature names")
    return X, names


def cv_permutation_importance(df: pd.DataFrame, best: pd.Series, n_repeats: int, n_folds=5, seed=42):
    """Refit (setting, best hyperparameters) on each of the same 5 GridSearchCV training folds used to select
    those hyperparameters, and measure permutation importance on each fold's held-out validation slice.

    PCA is refit per fold, so individual principal-component axes are not comparable across folds (fold 1's
    PC1 need not be fold 3's PC1) -- only the named (non-PCA) inorganic/water features are tracked individually
    across folds; the cation representation is tracked as one summed block-total per fold.
    """
    rdkit = ml.rdkit_descriptors(df.canonical_cation_smiles) if "rdkit" in best.setting else pd.DataFrame(index=df.index)
    smi = df[[c for c in df.columns if c.startswith("smi_ted_")]].reset_index(drop=True) if "smi_ted" in best.setting else pd.DataFrame(index=df.index)
    
    configs = [("halide_per_metal", ["inorganic_halide_per_metal"], pd.DataFrame(index=df.index), None, False)]
    for name, z, scale in [("rdkit", rdkit, True), ("smi_ted", smi, False)]:
        for label, variance in [("no_pca", None), ("pca80", .80), ("pca90", .90)]: configs.append((f"inorganic+{name}_{label}", ml.INORGANIC, z, variance, scale))
    _, common_cols, rep, variance, scale_rep = next(c for c in configs if c[0] == best.setting)

    common = df[common_cols].reset_index(drop=True)
    columns = common_cols + list(rep.columns)
    X = np.c_[common.to_numpy(float), rep.to_numpy(float)]
    y = df.target_non0D.astype(int).to_numpy()
    groups = df.connected_group.to_numpy()
    tr = np.flatnonzero(df.split.eq("train"))
    cv = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr], groups[tr]))
    best_params = json.loads(best.best_params)

    named_rows, cation_totals = [], []
    for fold, (fold_tr_rel, fold_val_rel) in enumerate(cv):
        fold_tr, fold_val = tr[fold_tr_rel], tr[fold_val_rel]
        features = ml.feature_pipeline(len(common_cols), rep.shape[1], variance, scale_rep)
        if best.model == "rf":
            pipe = Pipeline([("features", features), ("model", RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced_subsample"))])
        else:
            pipe = Pipeline([("features", features), ("scale", RobustScaler()), ("model", SVC(class_weight="balanced"))])
        pipe.set_params(**best_params)
        pipe.fit(X[fold_tr], y[fold_tr])

        val_df = df.iloc[fold_val].reset_index(drop=True)
        X_val, names = transformed_matrix_and_names(pipe, val_df, columns)
        downstream = pipe[1:]
        result = permutation_importance(downstream, X_val, y[fold_val], scoring="f1_macro", n_repeats=n_repeats, random_state=42, n_jobs=-1)

        is_cation = [n.startswith(("smi_ted", "rdkit")) for n in names]
        for name, imp, cat in zip(names, result.importances_mean, is_cation):
            if not cat: named_rows.append({"fold": fold, "feature": name, "importance": imp})
        cation_totals.append({"fold": fold, "feature": "cation_representation_total", "importance": float(np.sum(np.asarray(result.importances_mean)[is_cation]))})

    per_fold = pd.DataFrame(named_rows + cation_totals)
    agg = per_fold.groupby("feature").importance.agg(mean_importance="mean", std_importance="std", n_folds="count").reset_index()
    agg["se_importance"] = agg.std_importance / np.sqrt(agg.n_folds)
    agg["folds_positive"] = per_fold.groupby("feature").importance.apply(lambda s: int((s > 0).sum())).to_numpy()
    agg["block"] = agg.feature.map(lambda f: "cation" if f == "cation_representation_total" else "inorganic")
    return agg.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def plot_importance(d: pd.DataFrame, xerr_col: str, xlabel: str, title: str, out_path: Path, show_title: bool = True):
    d = d.iloc[::-1]
    labels = d.feature.map(lambda f: FEATURE_LABEL.get(f, f))
    fig, ax = new_fig((8.6, 7.4)) 
    colors = [CATION_COLOR if b == "cation" else INORGANIC_COLOR for b in d.block]
    ax.barh(range(len(d)), d.mean_importance, xerr=d[xerr_col], height=0.66, color=colors,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3}, zorder=3)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(labels, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=19)
    ax.tick_params(axis="x", labelsize=18)
    if show_title:
        ax.set_title(title, fontsize=17, loc="left", pad=12, linespacing=1.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=INORGANIC_COLOR), plt.Rectangle((0, 0), 1, 1, color=CATION_COLOR)]
    ax.legend(handles, ["Inorganic features", "Cation representation"], loc="center right", frameon=False, fontsize=17, handlelength=1.4, handleheight=1.0)
    ax.axvline(0, color="#c7c6bf", lw=1)
    style_axes(ax, grid_axis="x")
    save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prepared_data.xlsx")
    ap.add_argument("--results", default="results_ml/ml_results.csv")
    ap.add_argument("--out", default="results_ml")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--n-repeats", type=int, default=300)
    ap.add_argument("--cv-n-repeats", type=int, default=100)
    ap.add_argument("--cv-folds", type=int, default=5)
    a = ap.parse_args()

    results = pd.read_csv(a.results)
    best = results.sort_values("test_macro_f1", ascending=False).iloc[0]
    print(f"Best ML model: {best.model} / {best.setting}  (test_macro_f1={best.test_macro_f1:.4f})")

    bundle = joblib.load(best.saved_model)
    pipe, columns = bundle["model"], bundle["input_columns"]

    df = pd.read_excel(a.data)
    test = df[df.split.eq("test")].reset_index(drop=True)
    y_test = test.target_non0D.astype(int).to_numpy()

    X_test, names = transformed_matrix_and_names(pipe, test, columns)
    downstream = pipe[1:] 

    result = permutation_importance(downstream, X_test, y_test, scoring="f1_macro", n_repeats=a.n_repeats, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({
        "feature": names,
        "block": ["cation" if n.startswith(("smi_ted", "rdkit")) else "inorganic" for n in names],
        "mean_importance": result.importances_mean,
        "std_importance": result.importances_std,
    })

    imp["se_importance"] = imp.std_importance / np.sqrt(a.n_repeats)
    imp["z_score"] = imp.mean_importance / imp.se_importance.replace(0, np.nan)
    imp = imp.sort_values("mean_importance", ascending=False).reset_index(drop=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out / "permutation_importance.csv", index=False)
    print(f"\n=== Single test-set estimate (n_repeats={a.n_repeats}) ===")
    print(imp.head(a.top).round(5).to_string(index=False))
    grouped = imp.groupby("block").mean_importance.agg(["sum", "mean", "size"])
    print("\nBy block (sum / per-feature mean / count):")
    print(grouped.round(4).to_string())

    figures = Path(a.figures); figures.mkdir(parents=True, exist_ok=True)
    plot_importance(
        imp.head(a.top), "se_importance",
        "permutation importance",
        f"Permutation importance (single test set)\nbest ML model ({best.model}, {best.setting})",
        figures / "ml_permutation_importance.png",
    )
    plot_importance(
        imp.head(a.top), "se_importance",
        "permutation importance",
        "",
        figures / "ml_permutation_importance_no_title.png",
        show_title=False,
    )

    print(f"\n=== Cross-validated estimate ({a.cv_folds} folds x {a.cv_n_repeats} repeats each) ===")
    cv_imp = cv_permutation_importance(df, best, a.cv_n_repeats, n_folds=a.cv_folds)
    cv_imp.to_csv(out / "permutation_importance_cv.csv", index=False)
    print(cv_imp.round(5).to_string(index=False))
    print(f"\n('folds_positive' = how many of the {a.cv_folds} folds had positive importance -- {a.cv_folds}/{a.cv_folds} or 0/{a.cv_folds} means the sign is consistent across held-out data, not just across shuffles)")

    plot_importance(
        cv_imp, "se_importance",
        f"permutation importance  (mean over {a.cv_folds} held-out CV folds; error = ±1 SE across folds)",
        f"Cross-validated permutation importance\nbest ML model ({best.model}, {best.setting})",
        figures / "ml_permutation_importance_cv.png",
    )


if __name__ == "__main__":
    main()
