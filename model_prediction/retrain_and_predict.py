"""Retrain the selected SVM configuration on all 400 labeled compounds and predict new compounds.


Reuses feature_pipeline/save_bundle/INORGANIC from ``ml_training.train_models`` (identical pipeline:
impute -> [PCA for SVM] -> RobustScaler (SVM only) -> classifier), and grouped CV folds from the
same ``connected_group`` column prepared by ``ml_training.prepare_data`` (paper+cation union-find), so no
group leaks into a fold's validation set here either.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from ml_training import train_models as run_ml

INORGANIC = run_ml.INORGANIC
feature_pipeline = run_ml.feature_pipeline
save_bundle = run_ml.save_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared labeled workbook")
    parser.add_argument("--new-features", type=Path, required=True, help="Candidate feature CSV")
    parser.add_argument("--out", type=Path, default=Path("results_ml/new_compounds"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_excel(args.data)
    smi_cols = [c for c in df.columns if c.startswith("smi_ted_")]
    smi = df[smi_cols].reset_index(drop=True)
    common = df[INORGANIC].reset_index(drop=True)
    columns = INORGANIC + list(smi.columns)
    X = np.c_[common.to_numpy(float), smi.to_numpy(float)]
    y = df.target_non0D.astype(int).to_numpy()
    groups = df.connected_group.to_numpy()
    n = len(df)
    print(f"Training on all {n} labeled compounds ({n - y.sum()} 0D, {y.sum()} non-0D), "
          f"{len(np.unique(groups))} connected groups")

    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y, groups))

    models_dir = args.out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- SVM: inorganic + smi_ted, PCA80 ----
    features_svm = feature_pipeline(len(INORGANIC), smi.shape[1], 0.80, False)
    svm = Pipeline([("features", features_svm), ("scale", RobustScaler()), ("model", SVC(class_weight="balanced"))])
    svm_grid = [{"model__kernel": ["linear"], "model__C": [.1, 1, 10]},
                {"model__kernel": ["rbf"], "model__C": [1, 10], "model__gamma": ["scale"]}]
    gs_svm = GridSearchCV(svm, svm_grid, scoring="f1_macro", cv=cv, n_jobs=args.n_jobs, refit=True).fit(X, y)

    # Threshold tuning uses the same recipe as ml_training.train_models.tuned_svm, but the out-of-fold
    # predictions and final fit both use the full dataset
    best_svm = clone(gs_svm.best_estimator_)
    oof = cross_val_predict(best_svm, X, y, cv=cv, method="decision_function", n_jobs=args.n_jobs)
    platt = LogisticRegression(max_iter=1000).fit(oof.reshape(-1, 1), y)
    oof_p1 = platt.predict_proba(oof.reshape(-1, 1))[:, 1]
    candidates = np.linspace(0.05, 0.95, 181)
    threshold = float(max(candidates, key=lambda t: f1_score(y, (oof_p1 >= t).astype(int), average="macro")))
    best_svm.fit(X, y)
    path_svm = models_dir / "svm_inorganic+smi_ted_pca80_fulldata.joblib"
    save_bundle(path_svm, best_svm, columns, "inorganic+smi_ted_pca80", "svm",
                {"threshold": threshold, "platt_scaler": platt, "cv_macro_f1": gs_svm.best_score_,
                 "best_params": gs_svm.best_params_, "n_train": n, "trained_on": "full_dataset (train+test)"})
    print(f"SVM best params: {gs_svm.best_params_}  cv macro-F1: {gs_svm.best_score_:.4f}  threshold: {threshold:.3f}")

    # ---- predict on the new compounds ----
    feature_df = pd.read_csv(args.new_features)
    bundle = joblib.load(path_svm)
    model_name, cols = bundle["model_name"], bundle["input_columns"]
    missing = [c for c in cols if c not in feature_df.columns]
    if missing:
        raise ValueError(f"{path_svm.name}: missing columns {missing[:5]}...")
    X_new = feature_df[cols].to_numpy(float)
    fitted = bundle["model"]
    dec = fitted.decision_function(X_new)
    p1 = bundle["platt_scaler"].predict_proba(dec.reshape(-1, 1))[:, 1]
    pred = (p1 >= bundle["threshold"]).astype(int)
    rows = []
    for i in range(len(feature_df)):
        rows.append({
            "compound_id": feature_df.compound_id.iloc[i],
            "cation": feature_df.organic_component.iloc[i],
            "formula": feature_df.actual_formula_scxrd.iloc[i],
            "model": model_name,
            "threshold": bundle["threshold"],
            "prediction": "non-0D" if pred[i] == 1 else "0D",
            "P(non-0D)": round(float(p1[i]), 3),
        })

    out = pd.DataFrame(rows)
    out_path = args.out / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 220)
    print("\n=== Full-data-retrained predictions ===")
    print(out.to_string(index=False))
    print(f"\nSaved models to {models_dir}, predictions to {out_path}")


if __name__ == "__main__":
    main()
