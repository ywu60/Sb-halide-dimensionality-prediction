"""Retrain the best configuration of each model family (RF and SVM, ranked by test macro-F1 in
``ml_results.csv``) on all 400 labeled compounds and predict new compounds.

Reuses feature_pipeline/save_bundle/INORGANIC from ``ml_training.train_models`` (identical pipeline:
impute -> [PCA] -> RobustScaler (SVM only) -> classifier), and grouped CV folds from the
same ``connected_group`` column prepared by ``ml_training.prepare_data`` (paper+cation union-find), so no
group leaks into a fold's validation set here either. Hyperparameters are re-tuned on the full data with the
same grids, and the SVM decision threshold is re-tuned on full-data out-of-fold predictions. Only
inorganic+smi_ted settings are supported, since ``featurize_new_compounds`` builds SMI-TED (not RDKit) features.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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

PCA_VARIANCE = {"no_pca": None, "pca80": 0.80, "pca90": 0.90}


def best_per_family(results: pd.DataFrame) -> pd.DataFrame:
    """Best configuration of each model family (rf, svm) by test macro-F1."""
    return results.sort_values("test_macro_f1", ascending=False).groupby("model", sort=False).head(1).reset_index(drop=True)


def fit_full_data(model_name: str, setting: str, X, y, cv, columns: list[str], n_smi: int, models_dir: Path, n_jobs: int) -> Path:
    """Re-tune and refit one (model, setting) on all labeled data; returns the saved bundle path."""
    variance = PCA_VARIANCE[setting.removeprefix("inorganic+smi_ted_")]
    # Same Pipeline structure and grids as ml_training.train_models (SMI-TED block is not standardized)
    features = feature_pipeline(len(INORGANIC), n_smi, variance, False)
    if model_name == "rf":
        estimator = Pipeline([("features", features), ("model", RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced_subsample"))])
        grid = {"model__n_estimators": [300], "model__max_depth": [None, 10], "model__min_samples_leaf": [1, 3], "model__max_features": ["sqrt"]}
    else:
        estimator = Pipeline([("features", features), ("scale", RobustScaler()), ("model", SVC(class_weight="balanced"))])
        grid = [{"model__kernel": ["linear"], "model__C": [.1, 1, 10]},
                {"model__kernel": ["rbf"], "model__C": [1, 10], "model__gamma": ["scale"]}]
    gs = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=n_jobs, refit=True).fit(X, y)

    if model_name == "svm":
        # Threshold tuning uses the same recipe as ml_training.train_models.tuned_svm, but the out-of-fold
        # predictions and final fit both use the full dataset
        fitted = clone(gs.best_estimator_)
        oof = cross_val_predict(fitted, X, y, cv=cv, method="decision_function", n_jobs=n_jobs)
        platt = LogisticRegression(max_iter=1000).fit(oof.reshape(-1, 1), y)
        oof_p1 = platt.predict_proba(oof.reshape(-1, 1))[:, 1]
        candidates = np.linspace(0.05, 0.95, 181)
        threshold = float(max(candidates, key=lambda t: f1_score(y, (oof_p1 >= t).astype(int), average="macro")))
        fitted.fit(X, y)
        extra = {"threshold": threshold, "platt_scaler": platt}
    else:
        fitted = gs.best_estimator_; threshold = 0.5; extra = {"threshold": 0.5}
    path = models_dir / f"{model_name}_{setting}_fulldata.joblib"
    save_bundle(path, fitted, columns, setting, model_name,
                extra | {"cv_macro_f1": gs.best_score_, "best_params": gs.best_params_, "n_train": len(y), "trained_on": "full_dataset (train+test)"})
    print(f"{model_name.upper()} best params: {gs.best_params_}  cv macro-F1: {gs.best_score_:.4f}  threshold: {threshold:.3f}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared labeled workbook")
    parser.add_argument("--new-features", type=Path, required=True, help="Candidate feature CSV")
    parser.add_argument("--results", type=Path, default=Path("results_ml/ml_results.csv"), help="ml_results.csv from ml_training.train_models")
    parser.add_argument("--out", type=Path, default=Path("results_ml/new_compounds"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    best = best_per_family(pd.read_csv(args.results))
    for b in best.itertuples():
        if not b.setting.startswith("inorganic+smi_ted_"):
            raise ValueError(f"Best {b.model} setting {b.setting} is not inorganic+smi_ted_*; featurize_new_compounds only builds SMI-TED features")
        print(f"Best {b.model.upper()}: {b.setting} (test macro-F1 {b.test_macro_f1:.3f})")

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

    # ---- fit each family's best config on all data, then predict the new compounds ----
    feature_df = pd.read_csv(args.new_features)
    rows = []
    for b in best.itertuples():
        model_name, setting = b.model, b.setting
        path = fit_full_data(model_name, setting, X, y, cv, columns, smi.shape[1], models_dir, args.n_jobs)
        bundle = joblib.load(path)
        cols = bundle["input_columns"]
        missing = [c for c in cols if c not in feature_df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing[:5]}...")
        X_new = feature_df[cols].to_numpy(float)
        fitted = bundle["model"]
        if model_name == "svm":
            dec = fitted.decision_function(X_new)
            p1 = bundle["platt_scaler"].predict_proba(dec.reshape(-1, 1))[:, 1]
            pred = (p1 >= bundle["threshold"]).astype(int)
        else:
            class_pos = {int(c): i for i, c in enumerate(fitted.classes_)}
            p1 = fitted.predict_proba(X_new)[:, class_pos[1]]
            pred = fitted.predict(X_new).astype(int)
        for i in range(len(feature_df)):
            rows.append({
                "compound_id": feature_df.compound_id.iloc[i],
                "cation": feature_df.organic_component.iloc[i],
                "formula": feature_df.actual_formula_scxrd.iloc[i],
                "model": model_name,
                "setting": setting,
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
