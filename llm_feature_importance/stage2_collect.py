"""Collect the 120 valid fixed-category LLM rankings defined by the Stage 2 protocol."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from llm_feature_importance.stage2_config import (
    CONDITIONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WITH_DATA_OUTPUT_DIR,
    FEATURE_IDS,
    MAX_COMPLETION_TOKENS,
    MODELS,
    PRESENTATION_ORDERS,
    PROTOCOL_VERSION,
    REASONING_EFFORT,
    build_prompt,
    protocol_payload,
    protocol_sha256,
    write_protocol_files,
)
from llm_feature_importance.stage2_dataset import (
    DEFAULT_DATA_PATH,
    DatasetContext,
    build_dataset_contexts,
    protocol_dataset_metadata,
)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump())
        except (AttributeError, TypeError):
            # openai==1.55 combined with this environment's Pydantic can expose
            # model_dump() while failing internally on __pydantic_serializer__.
            pass
    if hasattr(value, "__dict__"):
        return {
            str(key): jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if hasattr(value, "dict"):
        try:
            return jsonable(value.dict())
        except (AttributeError, TypeError):
            pass
    return str(value)


def parse_ranking(raw_response: str) -> list[str]:
    """Strictly validate the complete JSON response and the F1-F7 permutation."""
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not a complete JSON document: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    if set(payload) != {"ranking"}:
        raise ValueError(f"JSON object must contain only 'ranking'; found keys {sorted(payload)}")
    ranking = payload["ranking"]
    if not isinstance(ranking, list) or len(ranking) != 7:
        raise ValueError("'ranking' must be a list of exactly seven feature IDs")
    if not all(isinstance(item, str) for item in ranking):
        raise ValueError("every ranking item must be a string feature ID")
    ranking = [item.strip() for item in ranking]
    if len(set(ranking)) != 7:
        raise ValueError(f"ranking contains a duplicate: {ranking}")
    expected = set(FEATURE_IDS)
    actual = set(ranking)
    if actual != expected:
        raise ValueError(
            f"ranking must contain F1-F7 exactly once; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return ranking


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
    return records


def verify_existing_records(
    records: list[dict],
    expected_protocol_hash: str,
    expected_condition: str = "no_data",
    dataset_contexts: dict[str, DatasetContext] | None = None,
) -> set[tuple[str, int]]:
    hashes = {record.get("protocol_sha256") for record in records}
    if hashes - {expected_protocol_hash}:
        raise RuntimeError(
            "Existing raw responses were created under a different protocol. Use a different output directory."
        )
    valid: dict[tuple[str, int], list[str]] = {}
    for record in records:
        # Records produced before condition support are the preserved no-data arm.
        record_condition = record.get("condition", "no_data")
        if record_condition != expected_condition:
            raise RuntimeError(
                f"Output directory mixes conditions: expected {expected_condition}, found {record_condition}"
            )
        if expected_condition == "with_data" and dataset_contexts is not None:
            model_id = str(record["model_id"])
            context = dataset_contexts.get(model_id)
            if context is None or record.get("dataset_sha256") != context.sha256:
                raise RuntimeError(f"Dataset hash mismatch in existing record for {model_id}")
        if not record.get("valid"):
            continue
        key = (str(record["model_id"]), int(record["trial_id"]))
        ranking = parse_ranking(record["raw_response"])
        if key in valid and ranking != valid[key]:
            raise RuntimeError(f"Conflicting valid responses already exist for {key}")
        valid[key] = ranking
    return set(valid)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_quota_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == "insufficient_quota" or "insufficient_quota" in str(exc)


def is_authentication_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in {401, 403} or type(exc).__name__ in {
        "AuthenticationError",
        "PermissionDeniedError",
    }


def collect_one(
    client: OpenAI,
    logger: JsonlLogger,
    model_id: str,
    model_name: str,
    trial_id: int,
    presentation_order: tuple[str, ...],
    protocol_hash: str,
    max_attempts: int,
    condition: str = "no_data",
    dataset_context: DatasetContext | None = None,
    paired_execution: dict | None = None,
) -> list[str]:
    if condition == "with_data" and dataset_context is None:
        raise ValueError("with_data collection requires a dataset context")
    if condition == "no_data" and dataset_context is not None:
        raise ValueError("no_data collection cannot receive a dataset context")
    dataset_block = dataset_context.block if dataset_context else None
    dataset_n_examples = dataset_context.n_examples if dataset_context else 0
    prompt = build_prompt(presentation_order, condition, dataset_block, dataset_n_examples)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    use_explicit_cache = condition == "with_data" and model_id.startswith("gpt-5.6")
    request_options = {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {"type": "json_object"},
        "reasoning_effort": REASONING_EFFORT,
    }
    if use_explicit_cache:
        marker = "\n\nBelow are the seven predefined features and their definitions."
        stable_prefix, dynamic_suffix = prompt.split(marker, 1)
        messages = [{"role": "user", "content": [
            {
                "type": "text",
                "text": stable_prefix,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {"type": "text", "text": marker + dynamic_suffix},
        ]}]
        request_options["messages"] = messages
        request_options["prompt_cache_key"] = f"stage2-with-data:{model_id}:{dataset_context.sha256}"
        request_options["prompt_cache_options"] = {"ttl": "30m"}
    settings = {
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {"type": "json_object"},
        "prompt_cache_mode": "explicit" if use_explicit_cache else "automatic",
    }
    dataset_metadata = dataset_context.call_metadata() if dataset_context else {
        "dataset_n_examples": 0,
        "dataset_sha256": None,
        "dataset_source": None,
        "dataset_source_sha256": None,
        "dataset_representation": None,
        "dataset_ordering_seed": None,
    }
    last_error = "unknown error"
    for retry_count in range(max_attempts):
        base_record = {
            "timestamp_utc": utc_now(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": protocol_hash,
            "condition": condition,
            "model_id": model_id,
            "model": model_name,
            "trial_id": trial_id,
            "presentation_order": list(presentation_order),
            "prompt": prompt,
            "retry_count": retry_count,
            "inference_settings": settings,
            "paired_execution": paired_execution,
        } | dataset_metadata
        try:
            response = client.chat.completions.create(**request_options)
            choice = response.choices[0]
            raw = choice.message.content or ""
            try:
                ranking = parse_ranking(raw)
                validation_error = None
                valid = True
            except ValueError as exc:
                ranking = None
                validation_error = str(exc)
                valid = False
            logger.append(
                base_record
                | {
                    "response_id": getattr(response, "id", None),
                    "response_model_id": getattr(response, "model", None),
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "raw_response": raw,
                    "parsed_ranking": ranking,
                    "valid": valid,
                    "validation_error": validation_error,
                    "usage": jsonable(getattr(response, "usage", None)),
                    "api_error": None,
                }
            )
            if valid:
                return ranking
            last_error = validation_error or "invalid response"
        except Exception as exc:
            logger.append(
                base_record
                | {
                    "response_id": None,
                    "response_model_id": None,
                    "finish_reason": None,
                    "raw_response": None,
                    "parsed_ranking": None,
                    "valid": False,
                    "validation_error": None,
                    "usage": None,
                    "api_error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            if is_quota_error(exc):
                raise RuntimeError(f"API quota exhausted while running {model_id}, trial {trial_id}") from exc
            if is_authentication_error(exc):
                raise RuntimeError(
                    f"API authentication/permission failure while running {model_id}, trial {trial_id}: {exc}"
                ) from exc
            last_error = f"{type(exc).__name__}: {exc}"
        if retry_count + 1 < max_attempts:
            time.sleep(min(2**retry_count, 20))
    raise RuntimeError(
        f"No valid response for {model_id}, trial {trial_id} after {max_attempts} attempts: {last_error}"
    )


def write_manifest(
    output_dir: Path,
    protocol_hash: str,
    condition: str = "no_data",
    dataset_metadata: dict | None = None,
) -> None:
    manifest = {
        "protocol_sha256": protocol_hash,
        "condition": condition,
        "protocol": protocol_payload(condition, dataset_metadata),
        "expected_valid_rankings": len(MODELS) * len(PRESENTATION_ORDERS),
        "expected_rankings_per_model": len(PRESENTATION_ORDERS),
    }
    path = output_dir / "run_manifest.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("run_manifest.json has a different protocol hash; use another output directory")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="no_data")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--models", nargs="+", choices=[m["model_id"] for m in MODELS])
    parser.add_argument("--trials", nargs="+", type=int, choices=range(1, 31))
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_attempts < 1:
        parser.error("--concurrency and --max-attempts must be positive")

    output_dir = args.out or (
        DEFAULT_WITH_DATA_OUTPUT_DIR if args.condition == "with_data" else DEFAULT_OUTPUT_DIR
    )
    dataset_contexts: dict[str, DatasetContext] | None = None
    dataset_metadata = None
    if args.condition == "with_data":
        dataset_contexts, dataset_examples = build_dataset_contexts(args.data)
        dataset_metadata = protocol_dataset_metadata(dataset_contexts)
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

    protocol_hash = protocol_sha256(args.condition, dataset_metadata)
    write_protocol_files(output_dir)
    write_manifest(output_dir, protocol_hash, args.condition, dataset_metadata)
    raw_path = output_dir / "raw_llm_responses.jsonl"
    raw_path.touch(exist_ok=True)
    if args.initialize_only:
        print(f"Initialized {args.condition} protocol in {output_dir}")
        return
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running the collection step")

    records = load_records(raw_path)
    complete = verify_existing_records(records, protocol_hash, args.condition, dataset_contexts)
    selected_models = [m for m in MODELS if args.models is None or m["model_id"] in args.models]
    selected_trials = sorted(set(args.trials or range(1, 31)))
    jobs = [
        (model, trial_id, PRESENTATION_ORDERS[trial_id - 1])
        for model in selected_models
        for trial_id in selected_trials
        if (model["model_id"], trial_id) not in complete
    ]
    requested_keys = {
        (model["model_id"], trial_id)
        for model in selected_models
        for trial_id in selected_trials
    }
    expected = len(requested_keys)
    print(f"Stage 2 protocol {protocol_hash[:12]}: {expected-len(complete & requested_keys)}/{expected} rankings to collect")
    if not jobs:
        print("All requested model/trial rankings are already complete.")
        return

    client = OpenAI(timeout=180, max_retries=0)
    logger = JsonlLogger(raw_path)
    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                collect_one,
                client,
                logger,
                model["model_id"],
                model["model"],
                trial_id,
                order,
                protocol_hash,
                args.max_attempts,
                args.condition,
                dataset_contexts[model["model_id"]] if dataset_contexts else None,
            ): (model, trial_id)
            for model, trial_id, order in jobs
        }
        completed_now = 0
        for future in as_completed(futures):
            model, trial_id = futures[future]
            try:
                ranking = future.result()
                completed_now += 1
                print(f"[{completed_now}/{len(jobs)}] {model['model']} trial {trial_id}: {ranking}", flush=True)
            except Exception as exc:
                failures.append((model["model_id"], trial_id, str(exc)))
                print(f"FAILED {model['model']} trial {trial_id}: {exc}", flush=True)

    final_complete = verify_existing_records(
        load_records(raw_path), protocol_hash, args.condition, dataset_contexts
    )
    missing = sorted(requested_keys - final_complete)
    if failures or missing:
        raise RuntimeError(f"Collection incomplete; {len(missing)} requested rankings remain missing: {missing}")
    print(f"Collection complete: {len(requested_keys)} valid requested rankings in {raw_path}")


if __name__ == "__main__":
    main()
