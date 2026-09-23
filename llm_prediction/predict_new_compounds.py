"""Predict dimensionality (0D vs non-0D) for new compounds using the selected
best LLM config from ``llm_prediction.evaluate_llms``: gpt-5.1-2025-11-13, all-shot, reasoning=medium,
representation=name (macro-F1 0.802 in results_llm/llm_results.csv -- the best LLM config overall,
beating representation=smiles's 0.773).

Reuses the evaluation module's serialization, prompt, response, and ordering functions so
the new compounds and the in-context examples are formatted identically to the original eval.

Unlike an ML retraining step, there is no "fit" to redo here -- an LLM prompt has no
trained parameters to refit. The equivalent of "use all labeled data, not just the 80% train
split" is using all 400 compounds (train+test combined) as the all-shot in-context examples,
instead of just the 321-row train split used for held-out evaluation.

gpt-5.1 is a reasoning model (supports_logprobs() is False for it), so it returns a hard 0/1
label only -- no calibrated probability, unlike the gpt-4.1 configs which had logprob-derived
probabilities available.

Set ``OPENAI_API_KEY`` directly or through the requested ``--env-file``.
"""
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODEL = "gpt-5.1-2025-11-13"
REASONING_EFFORT = "medium"
REASONING_MAX_TOKENS = 16384
SEED = 42


def load_env_file(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


from llm_prediction import evaluate_llms as run_llm


def build_new_rows(feature_path: Path):
    feature_df = pd.read_csv(feature_path)
    # prompt_for()/chemical_block() only touch these columns; everything else (smi_ted_*, etc.) is unused here.
    needed = ["compound_id", "organic_component", "canonical_cation_smiles", "formula_normalized",
              "sb_oxidation_state", "water_count", "inorganic_F_fraction", "inorganic_Cl_fraction",
              "inorganic_Br_fraction", "inorganic_I_fraction", "inorganic_halide_per_metal"]
    df = feature_df[needed + ["actual_formula_scxrd"]].copy()
    # CSV round-trip parsed "+3" as the integer 3, dropping the leading "+" that the training
    # shots' sb_oxidation_state strings ("+3", "+5", "+3/+5") always carry -- restore it so the
    # new compounds are formatted identically to the in-context examples.
    df["sb_oxidation_state"] = df["sb_oxidation_state"].apply(
        lambda v: v if str(v).startswith("+") else f"+{v}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared labeled workbook")
    parser.add_argument("--new-features", type=Path, required=True, help="Candidate feature CSV")
    parser.add_argument("--output", type=Path, default=Path("results_llm/new_compound_predictions.csv"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    load_env_file(args.env_file)
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY (env var or .env file next to this script) before running")
    client = OpenAI(timeout=180, max_retries=0)

    labeled = pd.read_excel(args.data)
    print(f"All-shot pool: {len(labeled)} labeled compounds (train+test combined, vs. evaluation's "
          f"321-row train-only pool)")
    all_shots = run_llm.order_all_shots(labeled, SEED, None)
    n_0d = int(all_shots.target_non0D.eq(0).sum())
    print(f"All-shot examples: {len(all_shots)} ({n_0d} 0D / {len(all_shots) - n_0d} non-0D)")

    new_rows = build_new_rows(args.new_features)
    encoding = run_llm.prompt_encoding()
    n_tokens = len(encoding.encode(run_llm.prompt_for(new_rows.iloc[0], "name", all_shots)))
    print(f"  representation=name: ~{n_tokens} prompt tokens/call x {len(new_rows)} compounds")

    rows = []
    lock = threading.Lock()

    def evaluate(job):
        representation, r = job
        messages = run_llm.prompt_messages(r, representation, all_shots)
        result = run_llm.one_token_result(
            client, args.model, messages, REASONING_EFFORT, REASONING_MAX_TOKENS
        )
        row = {
            "compound_id": r.compound_id, "cation": r.organic_component, "formula": r.actual_formula_scxrd,
            "model": args.model, "representation": representation, "prompting": "allshot_fulldata",
            "prediction": "non-0D" if result["prediction"] == 1 else "0D",
        }
        with lock:
            rows.append(row)
            print(f"[{len(rows)}/{len(new_rows)}] {representation} | {r.compound_id} | {r.organic_component} "
                  f"-> {row['prediction']}", flush=True)
        return row

    jobs = [("name", r) for _, r in new_rows.iterrows()]
    # Prime the shared all-shot prefix with one call first so the rest hit OpenAI's prompt cache
    # instead of all missing it simultaneously.
    evaluate(jobs[0])
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(evaluate, jobs[1:]))

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== LLM predictions ({args.model}, name representation, all-shot on {len(labeled)} labeled compounds) ===")
    print(out.sort_values("compound_id").to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
