"""Collect matched no-data/with-data trial pairs for the Stage 2 experiment."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from llm_feature_importance.stage2_collect import (
    JsonlLogger,
    collect_one,
    load_records,
    verify_existing_records,
    write_manifest,
)
from llm_feature_importance.stage2_config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WITH_DATA_OUTPUT_DIR,
    MODELS,
    PRESENTATION_ORDERS,
    protocol_sha256,
    write_protocol_files,
)
from llm_feature_importance.stage2_dataset import (
    DEFAULT_DATA_PATH,
    DatasetContext,
    build_dataset_contexts,
    protocol_dataset_metadata,
)


CONDITIONS = ("no_data", "with_data")
PAIRED_EXECUTION_VERSION = "stage2-paired-alternating-v1"


def paired_condition_order(trial_id: int) -> tuple[str, str]:
    """Counterbalance condition order: 15 trials begin with each condition."""
    if trial_id not in range(1, 31):
        raise ValueError(f"trial_id must be 1-30, got {trial_id}")
    return ("no_data", "with_data") if trial_id % 2 else ("with_data", "no_data")


def write_dataset_audits(output_dir: Path, dataset_examples) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_examples.to_csv(output_dir / "dataset_examples_by_representation.csv", index=False)
    dataset_order = (
        dataset_examples[
            ["example_position", "source_row_index", "compound_id", "target_non0D"]
        ]
        .drop_duplicates()
        .sort_values("example_position")
    )
    if len(dataset_order) != 321:
        raise RuntimeError(f"Expected 321 rows in the shared dataset order, found {len(dataset_order)}")
    dataset_order.to_csv(output_dir / "dataset_example_order.csv", index=False)


def initialize_conditions(
    no_data_dir: Path,
    with_data_dir: Path,
    data_path: Path,
) -> tuple[dict[str, DatasetContext], dict[str, str]]:
    if no_data_dir.resolve() == with_data_dir.resolve():
        raise RuntimeError("The paired conditions must use different output directories")
    contexts, dataset_examples = build_dataset_contexts(data_path)
    dataset_metadata = protocol_dataset_metadata(contexts)
    hashes = {
        "no_data": protocol_sha256("no_data"),
        "with_data": protocol_sha256("with_data", dataset_metadata),
    }
    for output_dir, condition in [(no_data_dir, "no_data"), (with_data_dir, "with_data")]:
        write_protocol_files(output_dir)
        write_manifest(
            output_dir,
            hashes[condition],
            condition,
            dataset_metadata if condition == "with_data" else None,
        )
        (output_dir / "raw_llm_responses.jsonl").touch(exist_ok=True)
    write_dataset_audits(with_data_dir, dataset_examples)

    execution_plan = {
        "paired_execution_version": PAIRED_EXECUTION_VERSION,
        "within_trial_calls_are_sequential": True,
        "order_rule": "odd trials: no_data then with_data; even trials: with_data then no_data",
        "trial_condition_order": {
            str(trial): list(paired_condition_order(trial)) for trial in range(1, 31)
        },
        "protocol_sha256_by_condition": hashes,
    }
    for output_dir in (no_data_dir, with_data_dir):
        (output_dir / "paired_execution_plan.json").write_text(
            json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8"
        )
    return contexts, hashes


def collect_pair(
    client: OpenAI,
    loggers: dict[str, JsonlLogger],
    model: dict,
    trial_id: int,
    contexts: dict[str, DatasetContext],
    hashes: dict[str, str],
    complete: dict[str, set[tuple[str, int]]],
    max_attempts: int,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, str]]]:
    key = (model["model_id"], trial_id)
    condition_order = paired_condition_order(trial_id)
    successes = []
    failures = []
    for pair_position, condition in enumerate(condition_order, start=1):
        if key in complete[condition]:
            continue
        paired_metadata = {
            "version": PAIRED_EXECUTION_VERSION,
            "condition_order": list(condition_order),
            "pair_position": pair_position,
        }
        try:
            ranking = collect_one(
                client=client,
                logger=loggers[condition],
                model_id=model["model_id"],
                model_name=model["model"],
                trial_id=trial_id,
                presentation_order=PRESENTATION_ORDERS[trial_id - 1],
                protocol_hash=hashes[condition],
                max_attempts=max_attempts,
                condition=condition,
                dataset_context=contexts[model["model_id"]] if condition == "with_data" else None,
                paired_execution=paired_metadata,
            )
            successes.append((condition, ranking))
        except Exception as exc:
            failures.append((condition, str(exc)))
    return successes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-data-out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--with-data-out", type=Path, default=DEFAULT_WITH_DATA_OUTPUT_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models", nargs="+", choices=[model["model_id"] for model in MODELS])
    parser.add_argument("--trials", nargs="+", type=int, choices=range(1, 31))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_attempts < 1:
        parser.error("--concurrency and --max-attempts must be positive")

    contexts, hashes = initialize_conditions(args.no_data_out, args.with_data_out, args.data)
    if args.initialize_only:
        print("Initialized paired no-data and with-data protocols; no API calls made.")
        return
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running the collection step")

    output_dirs = {"no_data": args.no_data_out, "with_data": args.with_data_out}
    loggers = {
        condition: JsonlLogger(output_dir / "raw_llm_responses.jsonl")
        for condition, output_dir in output_dirs.items()
    }
    complete = {
        condition: verify_existing_records(
            load_records(output_dir / "raw_llm_responses.jsonl"),
            hashes[condition],
            condition,
            contexts if condition == "with_data" else None,
        )
        for condition, output_dir in output_dirs.items()
    }
    selected_models = [
        model for model in MODELS if args.models is None or model["model_id"] in args.models
    ]
    selected_trials = sorted(set(args.trials or range(1, 31)))
    jobs = [
        (model, trial_id)
        for model in selected_models
        for trial_id in selected_trials
        if any((model["model_id"], trial_id) not in complete[condition] for condition in CONDITIONS)
    ]
    requested = {
        condition: {
            (model["model_id"], trial_id)
            for model in selected_models
            for trial_id in selected_trials
        }
        for condition in CONDITIONS
    }
    missing_calls = sum(len(requested[c] - complete[c]) for c in CONDITIONS)
    print(
        f"Paired Stage 2: {len(jobs)} trial pairs, {missing_calls} API calls remaining; "
        f"within-pair order alternates by trial."
    )
    if not jobs:
        print("All requested trial pairs are already complete.")
        return

    client = OpenAI(timeout=180, max_retries=0)
    failures = []
    completed_calls = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                collect_pair,
                client,
                loggers,
                model,
                trial_id,
                contexts,
                hashes,
                complete,
                args.max_attempts,
            ): (model, trial_id)
            for model, trial_id in jobs
        }
        for future in as_completed(futures):
            model, trial_id = futures[future]
            successes, pair_failures = future.result()
            for condition, ranking in successes:
                completed_calls += 1
                print(
                    f"[{completed_calls}/{missing_calls}] {model['model']} trial {trial_id} "
                    f"{condition}: {ranking}",
                    flush=True,
                )
            for condition, error in pair_failures:
                failures.append((model["model_id"], trial_id, condition, error))
                print(f"FAILED {model['model']} trial {trial_id} {condition}: {error}", flush=True)

    final_complete = {
        condition: verify_existing_records(
            load_records(output_dirs[condition] / "raw_llm_responses.jsonl"),
            hashes[condition],
            condition,
            contexts if condition == "with_data" else None,
        )
        for condition in CONDITIONS
    }
    missing = {
        condition: sorted(requested[condition] - final_complete[condition])
        for condition in CONDITIONS
    }
    if failures or any(missing.values()):
        raise RuntimeError(f"Paired collection incomplete; missing={missing}; failures={failures}")
    print(
        f"Paired collection complete: {len(requested['no_data'])} no-data and "
        f"{len(requested['with_data'])} with-data rankings."
    )


if __name__ == "__main__":
    main()
