"""Build inorganic + SMI-TED features for the 10 new mixed-halide Sb compounds, plus a hypothetical
"pure-halide" sibling of each (same cation, same Sb-halide framework size, but the entire inorganic
halide site collapsed onto whichever of Cl/I was already the majority)
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from ml_training import prepare_data as prep

CATION_SMILES = {
    "2-thiopheneethylammonium": "[NH3+]CCc1cccs1",
    "4-chlorophenethylammonium": "[NH3+]CCc1ccc(Cl)cc1",
    "morpholinium": "C1COCC[NH2+]1",
    "2-thiophenemethylammonium": "[NH3+]Cc1cccs1",
    "cyclohexylmethylammonium": "[NH3+]CC1CCCCC1",
    "4-fluorophenethylammonium": "[NH3+]CCc1ccc(F)cc1",
    "cyclohexylammonium": "[NH3+]C1CCCCC1",
    "iso-pentylammonium (methylbutylammonium)": "CC(C)CC[NH3+]",
}


def build_input_df(input_path: Path, sheet_name: str):
    new = pd.read_excel(input_path, sheet_name=sheet_name)
    df = pd.DataFrame({
        "compound_id": [f"NEW_{i+1:02d}" for i in range(len(new))],
        "paper_id": "NEW",
        "organic_component": new["A-cation"],
        "canonical_cation_smiles": new["A-cation"].map(CATION_SMILES),
        "formula_normalized": new["Actual formula (SCXRD)"],
        "water_count": 0.0,
        "Sb_halide": "Cl/I",
        "sb_oxidation_state": new["Sb charge"].str.replace("+", "", regex=False).radd("+"),
    })
    assert df["canonical_cation_smiles"].notna().all(), "Unmapped A-cation name(s) found"
    return df, new


def pure_halide_row(row: pd.Series, majority: str, minority: str) -> pd.Series:

    tokens = row.formula_normalized.split()
    parsed = [re.match(r"([A-Z][a-z]?)([0-9.]*)", tok) for tok in tokens]
    total = sum(float(m.group(2)) if m.group(2) else 1.0
                for m in parsed if m.group(1) in (majority, minority))
    n_str = str(int(total)) if float(total).is_integer() else str(total)
    new_tokens = []
    for tok, m in zip(tokens, parsed):
        el = m.group(1)
        if el == minority:
            continue
        new_tokens.append(f"{el}{n_str}" if el == majority else tok)
    new_row = row.copy()
    new_row["compound_id"] = f"{row.compound_id}_pure"
    new_row["formula_normalized"] = " ".join(new_tokens)
    new_row["Sb_halide"] = majority
    return new_row


def build_pure_halide_rows(mixed_df: pd.DataFrame, mixed_fractions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(mixed_df)):
        cl, iod = mixed_fractions.iloc[i][["inorganic_Cl_fraction", "inorganic_I_fraction"]]
        majority, minority = ("Cl", "I") if cl >= iod else ("I", "Cl")
        rows.append(pure_halide_row(mixed_df.iloc[i], majority, minority))
    return pd.DataFrame(rows).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Workbook containing the candidate compounds")
    parser.add_argument("--sheet", default="New Compounds")
    parser.add_argument("--smi-ted-dir", type=Path, required=True, help="Directory containing smi_ted_light/load.py and its checkpoint")
    parser.add_argument("--output", type=Path, default=Path("data/new_compound_features.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    mixed_df, raw = build_input_df(args.input, args.sheet)
    mixed_fractions = prep.common_features(mixed_df)
    pure_df = build_pure_halide_rows(mixed_df, mixed_fractions)

    df = pd.concat([mixed_df, pure_df], ignore_index=True)
    common = prep.common_features(df)
    df = pd.concat([df, common], axis=1)

    # The selected deployment model uses inorganic features plus SMI-TED embeddings.
    smi_feats = prep.smi_ted_embeddings(
        df.canonical_cation_smiles, str(args.smi_ted_dir), batch_size=args.batch_size
    ).reset_index(drop=True)

    feature_df = pd.concat([df.reset_index(drop=True), smi_feats], axis=1)

    feature_df["actual_formula_scxrd"] = pd.concat(
        [raw["Actual formula (SCXRD)"], pure_df["formula_normalized"]], ignore_index=True
    ).to_numpy()

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    print(f"Saved {feature_df.shape} feature matrix to {out_path} "
          f"(10 mixed-halide + 10 pure-halide siblings)")


if __name__ == "__main__":
    main()
