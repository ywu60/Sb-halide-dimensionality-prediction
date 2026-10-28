"""Reproduce Figure 3: LLM test macro-F1 across prompting conditions and cation representations.

Two panels (IUPAC name input, SMILES input), each with zero-shot, 12-shot, and all-shot bars per LLM and a
dashed reference line at the best ML model's test macro-F1. Drawn from the committed reference data
results_llm/llm_results.csv with the same plotting code as evaluation/plot_llm_results.py, which also writes
alternative/diagnostic views of the same results (not used in the manuscript).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.plot_llm_results import SEQUENTIAL_PALETTES, load_dataframe, plot_by_representation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-results", type=Path, default=REPOSITORY_ROOT / "results_llm" / "llm_results.csv",
                        help="LLM evaluation results (default: results_llm/llm_results.csv).")
    parser.add_argument("--gpt6-astra-results", type=Path,
                        default=REPOSITORY_ROOT / "results_llm" / "gpt6_astra_prompting_summary.csv",
                        help="Optional Astra macro-F1 summary to append, as in evaluation.plot_llm_results.")
    # Best ML test macro-F1 (RF, inorganic + SMI-TED PCA-80 in results_ml/ml_results.csv); same default as
    # evaluation.plot_llm_results, since results_ml/ is not committed.
    parser.add_argument("--ml-baseline", type=float, default=0.714, help="Best ML test macro-F1 for the dashed reference line.")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.llm_results.exists():
        raise FileNotFoundError(f"Missing {args.llm_results}. See the README for how to generate it.")
    df = pd.read_csv(args.llm_results)
    if args.gpt6_astra_results.exists():
        astra = pd.read_csv(args.gpt6_astra_results)
        df = pd.concat([df, astra.reindex(columns=df.columns)], ignore_index=True)
    df = load_dataframe(df)

    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        plot_by_representation(df, args.ml_baseline, args.output / f"figure_3.{suffix}",
                               SEQUENTIAL_PALETTES["selected_gray_blue_navy"])
    print(f"Saved Figure 3 to {args.output.resolve()}")


if __name__ == "__main__":
    main()
