"""Feature engineering + a single grouped train/test split for the analysis-3 dataset.

Adapted from analysis 2's 01_prepare_data.py for merged_compound_cif_cation.xlsx:
  - drops rows with unknown/missing dimensionality
  - target is FLIPPED relative to analysis 2: target_non0D = 0 for 0D, 1 for non-0D
    (analysis 2 used the opposite convention, 1 = 0D)
  - adds a water_count feature (new column in this dataset, not present in analysis 2)
  - train/test split groups are the union-find connected components of paper_id AND
    canonical_cation_smiles, so neither a paper nor a cation can appear on both sides
    of the split (analysis 2 only grouped by paper; here we also guard against cation leakage)

    python 01_prepare_data.py --input "../merged_compound_cif_cation.xlsx" --output prepared_data.xlsx
"""
import argparse, importlib, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

HALIDES = ["F", "Cl", "Br", "I"]
DIMS = ["0D", "1D", "2D", "3D"]

def parse_formula(s: str):
    s = re.sub(r"[\[\]()]", "", str(s).replace(" ", ""))
    if "x" in s.lower() or re.search(r"(?<=\d)[+-](?=\d)", s): return None
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", s)
    if not parts or "".join(e + n for e, n in parts) != s: return None
    counts = {}
    for e, n in parts: counts[e] = counts.get(e, 0.0) + (float(n) if n else 1.0)
    return counts

def cation_counts(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: raise ValueError(f"Invalid cation SMILES: {smiles}")
    counts = {}
    for atom in mol.GetAtoms(): counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0.0) + 1.0
    return counts

def cation_multiplier(total: dict, organic: dict):
    ratios = [total[e] / organic[e] for e in ["C", "N", "S", "P"] if organic.get(e, 0) > 0 and total.get(e, 0) > 0]
    if not ratios: return 1.0
    rounded = [round(x) for x in ratios if abs(x - round(x)) < .08]
    return float(np.median(rounded or ratios))

def inorganic_composition(r: pd.Series, total: dict | None):
    """Return (halide fractions, inorganic-halide-per-metal ratio, mixed flag) for the inorganic sublattice only.

    These describe the Sb/Bi-halide anion framework (composition + X:M stoichiometry) and deliberately
    contain no whole-formula organic composition, so they can serve as a cation-free baseline."""
    listed = [x.strip() for x in str(r.Sb_halide).split("/")]
    if any(x not in HALIDES for x in listed): raise ValueError(f"Unsupported Sb_halide for {r.compound_id}: {r.Sb_halide}")
    mixed = int(len(listed) > 1)
    metal = None if total is None else (total.get("Sb", 0.0) + total.get("Bi", 0.0))
    if not mixed:
        fractions = {h: float(h == listed[0]) for h in HALIDES}
        if total is None or not metal: return fractions, np.nan, mixed
        organic = cation_counts(r.canonical_cation_smiles); mult = cation_multiplier(total, organic)
        inorg_x = max(total.get(listed[0], 0.0) - mult * organic.get(listed[0], 0.0), 0.0)
        return fractions, (inorg_x / metal if inorg_x > 0 else np.nan), mixed
    if total is None: raise ValueError(f"Variable formula cannot determine mixed Sb-halide fractions for {r.compound_id}")
    organic = cation_counts(r.canonical_cation_smiles); mult = cation_multiplier(total, organic)
    inorganic = {h: max(total.get(h, 0.0) - mult * organic.get(h, 0.0), 0.0) if h in listed else 0.0 for h in HALIDES}
    denom = sum(inorganic.values())
    if denom <= 0: raise ValueError(f"Cannot determine inorganic halides for {r.compound_id}")
    unexpected = {h: total.get(h, 0.0) - mult * organic.get(h, 0.0) for h in HALIDES if h not in listed}
    if any(v > .08 for v in unexpected.values()):
        raise ValueError(f"Formula has unexplained halide outside Sb_halide for {r.compound_id}: {unexpected}")
    fractions = {h: inorganic[h] / denom for h in HALIDES}
    return fractions, (denom / metal if metal else np.nan), mixed

def common_features(df: pd.DataFrame):
    """Model-input features only: Sb-oxidation-state one-hot and the inorganic Sb/Bi-halide sublattice
    composition (used by 03_run_ml.py's INORGANIC block). Whole-formula organic composition (elemental
    fractions, H/C and N/C ratios, molecular weight, Bi_fraction, has_water, ...) is deliberately not
    computed here: no downstream script consumes it, and the ML feature design keeps organic/cation
    information entirely out of this block (see 03_run_ml.py's INORGANIC comment)."""
    rows = []
    for _, r in df.iterrows():
        c = parse_formula(r.formula_normalized)
        ox = str(r.sb_oxidation_state).strip()
        x = {"sb_III": int(ox == "+3"), "sb_V": int(ox == "+5"), "sb_mixed_valence": int("/" in ox)}
        fractions, x_per_metal, mixed = inorganic_composition(r, c)
        x |= {f"inorganic_{h}_fraction": fractions[h] for h in HALIDES}
        x["mixed_inorganic_halide"] = mixed
        x["inorganic_halide_per_metal"] = x_per_metal
        rows.append(x)
    return pd.DataFrame(rows, index=df.index)

def connected_groups(df: pd.DataFrame):
    """Union-find over (paper_id, canonical_cation_smiles): a paper and every cation it reports are fused
    into one component, and two papers that happen to share a cation are fused into the same component too.
    This is stricter than analysis 2 (which grouped by paper only, letting a cation recur across the split) --
    here neither a paper nor a cation can appear on both sides of train/test."""
    parent: dict = {}
    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root: root = parent[root]
        while parent[x] != root: parent[x], x = root, parent[x]
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for _, r in df.iterrows():
        union(("paper", str(r.paper_id)), ("cation", str(r.canonical_cation_smiles)))
    roots = [find(("paper", str(r.paper_id))) for _, r in df.iterrows()]
    mapping = {root: i for i, root in enumerate(dict.fromkeys(roots))}
    return np.array([mapping[root] for root in roots])

def grouped_holdout(y: np.ndarray, groups: np.ndarray, test_size=.20, seed=42, tries=50000):
    rng = np.random.default_rng(seed); unique = np.unique(groups); target_n = round(len(y) * test_size); target_rate = y.mean(); best = None
    members = {g: np.flatnonzero(groups == g) for g in unique}
    for _ in range(tries):
        chosen, n = [], 0
        for g in rng.permutation(unique):
            if n + len(members[g]) <= target_n or n < target_n * .75: chosen.append(g); n += len(members[g])
            if n >= target_n: break
        te = np.flatnonzero(np.isin(groups, chosen))
        if len(np.unique(y[te])) < 2 or len(np.unique(y[np.setdiff1d(np.arange(len(y)), te)])) < 2: continue
        score = abs(len(te) - target_n) / len(y) + abs(y[te].mean() - target_rate)
        if best is None or score < best[0]: best = score, te
    if best is None: raise RuntimeError("Could not construct a valid grouped split")
    te = best[1]; return np.setdiff1d(np.arange(len(y)), te), te

def smi_ted_directory(local_dir: str | None):
    if local_dir:
        roots = [Path(local_dir)]
    else:
        from huggingface_hub import hf_hub_download, snapshot_download

        root = Path(snapshot_download(
            repo_id="ibm-research/materials.smi-ted",
            allow_patterns=["smi-ted/inference/smi_ted_light/*"],
        ))
        roots = [root]

    candidates = []
    for root in roots:
        if (root / "load.py").exists():
            candidates.append(root)
        candidates += list(root.glob("**/smi_ted_light"))

    candidates = [p for p in candidates if (p / "load.py").exists()]
    if not candidates:
        raise FileNotFoundError(
            "Could not find smi_ted_light/load.py. "
            "Supply --smi-ted-dir."
        )

    model_dir = candidates[0]
    checkpoints = list(model_dir.glob("*.pt"))

    if not checkpoints and not local_dir:
        hf_hub_download(
            repo_id="ibm-research/materials.smi-ted",
            filename="smi-ted-Light_40.pt",
            local_dir=model_dir,
        )

    return model_dir

def smi_ted_embeddings(smiles: pd.Series, local_dir: str | None, batch_size: int):
    import torch
    model_dir = smi_ted_directory(local_dir); sys.path.insert(0, str(model_dir.parent))
    load = importlib.import_module("smi_ted_light.load").load_smi_ted
    checkpoints = sorted(model_dir.glob("*.pt"))
    if not checkpoints: raise FileNotFoundError(f"No SMI-TED .pt checkpoint in {model_dir}")
    preferred = next((p for p in checkpoints if p.name == "smi_ted_light.pt"), checkpoints[0])
    model = load(folder=str(model_dir), ckpt_filename=preferred.name); blocks = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            z = model.encode(smiles.iloc[start:start + batch_size].reset_index(drop=True), return_torch=True)
            blocks.append(z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z))
    z = np.vstack(blocks)
    return pd.DataFrame(z, columns=[f"smi_ted_{i:04d}" for i in range(z.shape[1])])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output", default="prepared_data.xlsx")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--smi-ted-dir"); ap.add_argument("--smi-batch-size", type=int, default=64); ap.add_argument("--skip-smi-ted", action="store_true"); a = ap.parse_args()
    df = pd.read_excel(a.input)
    df = df.rename(columns={"Paper ID": "paper_id", "Water_count": "water_count"})
    required = {"compound_id", "paper_id", "formula_normalized", "water_count", "Sb_halide", "sb_oxidation_state", "dimensionality", "organic_component", "canonical_cation_smiles"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Input columns missing: {sorted(missing)}")

    n_before = len(df)
    df["dimensionality"] = df.dimensionality.astype(str).str.strip()
    df = df[df.dimensionality.isin(DIMS)].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with unknown/missing dimensionality ({n_before} -> {len(df)})")

    # FLIPPED vs. analysis 2: 0 = 0D, 1 = non-0D.
    df["target_non0D"] = (df.dimensionality != "0D").astype(int)
    groups = connected_groups(df)
    tr, te = grouped_holdout(df.target_non0D.to_numpy(), groups, seed=a.seed)
    df["connected_group"] = groups; df["split"] = "train"; df.loc[te, "split"] = "test"
    result = pd.concat([df, common_features(df)], axis=1)
    if not a.skip_smi_ted: result = pd.concat([result, smi_ted_embeddings(df.canonical_cation_smiles, a.smi_ted_dir, a.smi_batch_size)], axis=1)

    assert not set(result.loc[tr, "paper_id"]) & set(result.loc[te, "paper_id"]), "paper_id leaked across split"
    assert not set(result.loc[tr, "canonical_cation_smiles"]) & set(result.loc[te, "canonical_cation_smiles"]), "cation leaked across split"

    Path(a.output).parent.mkdir(parents=True, exist_ok=True); result.to_excel(a.output, index=False)
    print(f"Saved {a.output} with {sum(c.startswith('smi_ted_') for c in result.columns)} SMI-TED features")
    print(result.groupby(["split", "target_non0D"]).size().unstack(fill_value=0))
    print(f"Train: {len(tr)}, test: {len(te)}, groups: {len(np.unique(groups))}, paper overlap: 0, cation overlap: 0")

if __name__ == "__main__": main()
