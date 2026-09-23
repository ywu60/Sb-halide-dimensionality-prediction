"""Audit every metered Stage 2 API attempt and estimate standard API cost."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_feature_importance.stage2_collect import load_records
from llm_feature_importance.stage2_config import DEFAULT_OUTPUT_DIR, MODEL_BY_ID, MODEL_IDS


# Official standard, short-context text-token prices checked 2026-09-09.
# Reasoning tokens are included in completion_tokens and therefore use the output rate.
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICING_USD_PER_MILLION = {
    "gpt-5.1-2025-11-13": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.5-2026-04-23": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-sol": {"input": 4.00, "cached_input": 0.40, "output": 20.00},
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
}


def usage_row(record: dict) -> dict:
    usage = record.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(prompt_details.get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    model_id = str(record["model_id"])
    price = PRICING_USD_PER_MILLION[model_id]
    estimated_cost = (
        (prompt - cached) * price["input"]
        + cached * price["cached_input"]
        + completion * price["output"]
    ) / 1_000_000
    return {
        "model": MODEL_BY_ID[model_id]["model"],
        "model_id": model_id,
        "trial": int(record["trial_id"]),
        "condition": record.get("condition", "no_data"),
        "retry_count": int(record.get("retry_count") or 0),
        "valid": bool(record.get("valid")),
        "finish_reason": record.get("finish_reason"),
        "validation_error": record.get("validation_error"),
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": prompt - cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "visible_completion_tokens": completion - reasoning,
        "total_tokens": int(usage.get("total_tokens") or prompt + completion),
        "estimated_standard_cost_usd": estimated_cost,
        "input_usd_per_million": price["input"],
        "cached_input_usd_per_million": price["cached_input"],
        "output_usd_per_million": price["output"],
        "pricing_source": PRICING_SOURCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = load_records(args.out / "raw_llm_responses.jsonl")
    relevant = [record for record in records if record.get("model_id") in MODEL_IDS]
    valid = [record for record in relevant if record.get("valid")]
    seen = set()
    for record in valid:
        key = (str(record["model_id"]), int(record["trial_id"]))
        if key in seen:
            raise RuntimeError(f"Duplicate valid response prevents unambiguous trial accounting: {key}")
        seen.add(key)
    call_rows = [usage_row(record) for record in relevant if record.get("usage")]
    if not call_rows:
        raise RuntimeError("No metered Stage 2 API attempts are available for token accounting")
    calls = pd.DataFrame(call_rows).sort_values(
        ["model_id", "trial", "retry_count"]
    ).reset_index(drop=True)

    unmetered_by_model = {}
    for model_id in MODEL_IDS:
        unmetered_by_model[model_id] = sum(
            record.get("model_id") == model_id
            and not record.get("usage")
            and not record.get("valid")
            for record in records
        )

    summary_rows = []
    for model_id, group in calls.groupby("model_id", sort=False):
        valid_group = group[group.valid]
        invalid_group = group[~group.valid]
        exact_cost = float(group.estimated_standard_cost_usd.sum())
        valid_cost = float(valid_group.estimated_standard_cost_usd.sum())
        invalid_cost = float(invalid_group.estimated_standard_cost_usd.sum())
        unmetered = int(unmetered_by_model[model_id])
        # Attempts with no usage object failed before a billable model response
        # was returned (for example, an invalid cache option). Count them for
        # auditability but do not assign token cost.
        overhead_estimate = 0.0
        summary_rows.append(
            {
                "model": MODEL_BY_ID[model_id]["model"],
                "model_id": model_id,
                "metered_attempts": len(group),
                "valid_calls": len(valid_group),
                "invalid_metered_attempts": len(invalid_group),
                "prompt_tokens": int(group.prompt_tokens.sum()),
                "cached_prompt_tokens": int(group.cached_prompt_tokens.sum()),
                "completion_tokens": int(group.completion_tokens.sum()),
                "reasoning_tokens": int(group.reasoning_tokens.sum()),
                "visible_completion_tokens": int(group.visible_completion_tokens.sum()),
                "total_tokens": int(group.total_tokens.sum()),
                "mean_total_tokens_per_metered_attempt": float(group.total_tokens.mean()),
                "estimated_valid_call_cost_usd": valid_cost,
                "estimated_invalid_attempt_cost_usd": invalid_cost,
                "estimated_all_metered_attempts_cost_usd": exact_cost,
                "mean_cost_per_metered_attempt_usd": float(group.estimated_standard_cost_usd.mean()),
                "unmetered_error_attempts": unmetered,
                "estimated_unmetered_error_cost_usd": overhead_estimate,
                "estimated_total_cost_including_unmetered_estimate_usd": exact_cost + overhead_estimate,
                "input_usd_per_million": float(group.input_usd_per_million.iloc[0]),
                "cached_input_usd_per_million": float(group.cached_input_usd_per_million.iloc[0]),
                "output_usd_per_million": float(group.output_usd_per_million.iloc[0]),
                "pricing_source": PRICING_SOURCE,
            }
        )
    summary = pd.DataFrame(summary_rows)
    calls.to_csv(args.out / "token_usage_by_call.csv", index=False)
    summary.to_csv(args.out / "token_cost_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out / 'token_usage_by_call.csv'} and {args.out / 'token_cost_summary.csv'}")


if __name__ == "__main__":
    main()
