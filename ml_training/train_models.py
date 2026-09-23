"""RandomForest + SVM baselines on the analysis-3 Sb-halide dataset.

Adapted from analysis 2's 02_run_ml.py:
  - target is target_non0D (0 = 0D, 1 = non-0D) -- FLIPPED vs. analysis 2's target_0D (1 = 0D)
  - the inorganic feature block gains water_count (new column in this dataset)
  - probability columns are named probability_0D / probability_non0D to match the new label meaning

    python 03_run_ml.py --data prepared_data.xlsx --out results_ml
"""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

# Cation-free baseline block: inorganic Sb-halide sublattice (anion composition, X:metal stoichiometry,
# oxidation state) plus water-of-crystallization count. Excludes whole-formula organic composition
# (fraction_C/H/N, H_C_ratio, molecular weight, ...) so that adding RDKit / SMI-TED cation features is a
# clean test of "does cation information help". Bi_fraction is also excluded: only 7/400 compounds have
# any Bi occupancy, too rare to support as a standalone feature.
INORGANIC = ["sb_III", "sb_V", "sb_mixed_valence", "inorganic_F_fraction", "inorganic_Cl_fraction", "inorganic_Br_fraction", "inorganic_I_fraction", "mixed_inorganic_halide", "inorganic_halide_per_metal", "water_count"]

def rdkit_descriptors(smiles: pd.Series):
    names = [n for n, _ in Descriptors._descList]; funcs = [f for _, f in Descriptors._descList]; rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None: raise ValueError(f"Invalid SMILES: {s}")
        vals = []
        for f in funcs:
            try: v = f(mol); vals.append(v if np.isfinite(v) else np.nan)
            except Exception: vals.append(np.nan)
        rows.append(vals)
    return pd.DataFrame(rows, columns=["rdkit_" + n for n in names])

def feature_pipeline(n_common: int, n_rep: int, variance: float | None, scale_rep: bool):
    common = Pipeline([("impute", SimpleImputer(strategy="median"))])
    if not n_rep: return ColumnTransformer([("common", common, list(range(n_common)))], remainder="drop")
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_rep: steps.append(("scale", StandardScaler()))
    if variance is not None: steps.append(("pca", PCA(n_components=variance, svd_solver="full")))
    return ColumnTransformer([("common", common, list(range(n_common))), ("representation", Pipeline(steps), list(range(n_common, n_common + n_rep)))], remainder="drop")

def save_bundle(path: Path, model, columns: list[str], setting: str, model_name: str, extra: dict | None = None):
    d = {"model": model, "input_columns": columns, "setting": setting, "model_name": model_name, "class_order": [0, 1], "target_definition": "0=0D; 1=non-0D"}
    if extra: d |= extra
    joblib.dump(d, path)

def tuned_svm(gs, X, y, tr, te, cv, n_jobs):
    """Refit the best SVM, then choose the decision threshold that maximizes macro-F1 on grouped, leak-free
    out-of-fold predictions (the default 0.5 threshold made the SVM collapse to the majority class). Returns
    test-set predictions, P(0D)/P(non-0D) from a Platt sigmoid, and the chosen threshold."""
    best = clone(gs.best_estimator_)
    oof = cross_val_predict(best, X[tr], y[tr], cv=cv, method="decision_function", n_jobs=n_jobs)
    platt = LogisticRegression(max_iter=1000).fit(oof.reshape(-1, 1), y[tr])
    oof_p1 = platt.predict_proba(oof.reshape(-1, 1))[:, 1]
    candidates = np.linspace(0.05, 0.95, 181)
    threshold = float(max(candidates, key=lambda t: f1_score(y[tr], (oof_p1 >= t).astype(int), average="macro")))
    best.fit(X[tr], y[tr])
    p1 = platt.predict_proba(best.decision_function(X[te]).reshape(-1, 1))[:, 1]
    pred = (p1 >= threshold).astype(int)
    return best, pred, 1.0 - p1, p1, threshold, platt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results_ml"); ap.add_argument("--n-jobs", type=int, default=-1); a = ap.parse_args()
    out = Path(a.out); models_dir = out / "models"; models_dir.mkdir(parents=True, exist_ok=True); df = pd.read_excel(a.data)
    missing = set(INORGANIC + ["compound_id", "canonical_cation_smiles", "target_non0D", "connected_group", "split"]) - set(df.columns)
    if missing: raise ValueError(f"Prepared-data columns missing: {sorted(missing)}. Run ml_training.prepare_data first.")
    smi_cols = [c for c in df.columns if c.startswith("smi_ted_")]
    if not smi_cols: raise ValueError("No smi_ted_* columns found. Run preparation without --skip-smi-ted.")
    rdkit = rdkit_descriptors(df.canonical_cation_smiles); smi = df[smi_cols].reset_index(drop=True)
    y = df.target_non0D.astype(int).to_numpy(); groups = df.connected_group.to_numpy(); tr = np.flatnonzero(df.split.eq("train")); te = np.flatnonzero(df.split.eq("test"))
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(tr)), y[tr], groups[tr]))
    # Baseline = the single dominant inorganic feature (halide-per-metal ratio), no cation, no water, evaluated
    # with the same RF/SVM models as every other setting so it is directly comparable. The cation-augmented
    # settings add a cation representation (RDKit descriptors / SMI-TED embeddings) on top of the full
    # inorganic + water block.
    configs = [("halide_per_metal", ["inorganic_halide_per_metal"], pd.DataFrame(index=df.index), None, False)]
    for name, z, scale in [("rdkit", rdkit, True), ("smi_ted", smi, False)]:
        for label, variance in [("no_pca", None), ("pca80", .80), ("pca90", .90)]: configs.append((f"inorganic+{name}_{label}", INORGANIC, z, variance, scale))
    results, predictions = [], []
    for setting, common_cols, rep, variance, scale_rep in configs:
        common = df[common_cols].reset_index(drop=True); columns = common_cols + list(rep.columns); X = np.c_[common.to_numpy(float), rep.to_numpy(float)]; features = feature_pipeline(len(common_cols), rep.shape[1], variance, scale_rep)
        rf = Pipeline([("features", features), ("model", RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced_subsample"))])
        rf_grid = {"model__n_estimators": [300], "model__max_depth": [None, 10], "model__min_samples_leaf": [1, 3], "model__max_features": ["sqrt"]}
        # SVM needs feature scaling. RobustScaler (median/IQR), not StandardScaler (mean/std): a handful of
        # training compounds are trace-metal-doped hosts (Sb occupancy near 0) whose inorganic_halide_per_metal
        # blows up to >1000 (halide count over a ~0 Sb+Bi denominator). StandardScaler's std is dominated by
        # those few outliers, which squashes every normal-range value (4-8) to near-zero after scaling and made
        # the feature look unimportant under permutation despite having the single largest SVM coefficient.
        # RobustScaler is insensitive to that handful of outliers, so the normal-range values keep real spread.
        # Its decision threshold is tuned (see tuned_svm) so it no longer collapses to the majority class as it
        # did with the default 0.5 cutoff.
        svm = Pipeline([("features", features), ("scale", RobustScaler()), ("model", SVC(class_weight="balanced"))])
        svm_grid = [{"model__kernel": ["linear"], "model__C": [.1, 1, 10]}, {"model__kernel": ["rbf"], "model__C": [1, 10], "model__gamma": ["scale"]}]
        for model_name, estimator, grid in [("rf", rf, rf_grid), ("svm", svm, svm_grid)]:
            gs = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=a.n_jobs, refit=True).fit(X[tr], y[tr])
            if model_name == "svm":
                fitted, pred, p0, p1, threshold, platt = tuned_svm(gs, X, y, tr, te, cv, a.n_jobs)
                extra = {"threshold": threshold, "platt_scaler": platt}
            else:
                fitted = gs.best_estimator_; threshold = 0.5; extra = {"threshold": 0.5}
                prob = fitted.predict_proba(X[te]); pred = fitted.predict(X[te]); class_pos = {int(c): i for i, c in enumerate(fitted.classes_)}; p0, p1 = prob[:, class_pos[0]], prob[:, class_pos[1]]
            model_path = models_dir / f"{model_name}_{setting}.joblib"; save_bundle(model_path, fitted, columns, setting, model_name, extra)
            rep_step = fitted.named_steps["features"].named_transformers_.get("representation")
            pca = rep_step.named_steps.get("pca") if hasattr(rep_step, "named_steps") else None; n_rep = int(pca.n_components_) if pca is not None else rep.shape[1]
            results.append({"model": model_name, "setting": setting, "n_rep_features": n_rep, "threshold": round(threshold, 3), "cv_macro_f1": gs.best_score_, "test_accuracy": accuracy_score(y[te], pred), "test_macro_f1": f1_score(y[te], pred, average="macro"), "best_params": json.dumps(gs.best_params_), "saved_model": str(model_path)})
            predictions += [{"compound_id": df.compound_id.iloc[i], "model": model_name, "setting": setting, "true": int(y[i]), "prediction": int(yhat), "probability_0D": float(q0), "probability_non0D": float(q1)} for i, yhat, q0, q1 in zip(te, pred, p0, p1)]
    result = pd.DataFrame(results); result.to_csv(out / "ml_results.csv", index=False); pd.DataFrame(predictions).to_csv(out / "ml_test_predictions.csv", index=False)
    print(result.sort_values(["model", "test_macro_f1"], ascending=[True, False]).to_string(index=False))

if __name__ == "__main__": main()
