"""Validate Stage 2 responses and produce all requested statistics and CSV tables."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from llm_feature_importance.stage2_collect import load_records, parse_ranking, verify_existing_records
from llm_feature_importance.stage2_config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEFAULT_OUTPUT_DIR,
    FEATURE_BY_ID,
    FEATURE_IDS,
    MODELS,
    MODEL_BY_ID,
    MODEL_IDS,
    PRESENTATION_ORDERS,
    write_protocol_files,
)


def select_complete_valid_records(
    records: list[dict],
    expected_protocol_hash: str | None = None,
    expected_condition: str = "no_data",
) -> dict[tuple[str, int], dict]:
    """Return one valid record per expected model/trial and reject ambiguous datasets."""
    if expected_protocol_hash is None:
        hashes = {record.get("protocol_sha256") for record in records}
        if len(hashes) != 1 or None in hashes:
            raise RuntimeError(f"Expected one recorded protocol hash, found {hashes}")
        expected_protocol_hash = next(iter(hashes))
    verify_existing_records(records, expected_protocol_hash, expected_condition)
    selected: dict[tuple[str, int], dict] = {}
    for record in records:
        if not record.get("valid"):
            continue
        key = (str(record["model_id"]), int(record["trial_id"]))
        if key not in {(model_id, trial) for model_id in MODEL_IDS for trial in range(1, 31)}:
            continue
        ranking = parse_ranking(record["raw_response"])
        if record.get("parsed_ranking") not in (None, ranking):
            raise RuntimeError(f"Stored parsed ranking disagrees with raw response for {key}")
        if tuple(record.get("presentation_order", ())) != PRESENTATION_ORDERS[key[1] - 1]:
            raise RuntimeError(f"Stored presentation order disagrees with protocol for {key}")
        if key in selected:
            prior = parse_ranking(selected[key]["raw_response"])
            if prior != ranking:
                raise RuntimeError(f"Conflicting valid rankings for {key}")
            raise RuntimeError(f"Duplicate valid ranking records for {key}; keep exactly one valid call")
        selected[key] = record
    expected = {(model_id, trial) for model_id in MODEL_IDS for trial in range(1, 31)}
    missing = sorted(expected - set(selected))
    if missing:
        counts = {
            model_id: sum((model_id, trial) in selected for trial in range(1, 31))
            for model_id in MODEL_IDS
        }
        raise RuntimeError(f"Expected 120 valid rankings; missing {len(missing)}. Valid counts: {counts}")
    return selected


def build_rankings_long(selected: dict[tuple[str, int], dict]) -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        model_name = MODEL_BY_ID[model_id]["model"]
        for trial in range(1, 31):
            ranking = parse_ranking(selected[(model_id, trial)]["raw_response"])
            order = PRESENTATION_ORDERS[trial - 1]
            rank_by_feature = {feature_id: rank for rank, feature_id in enumerate(ranking, start=1)}
            position_by_feature = {feature_id: pos for pos, feature_id in enumerate(order, start=1)}
            for feature_id in FEATURE_IDS:
                rank = rank_by_feature[feature_id]
                rows.append(
                    {
                        "model": model_name,
                        "model_id": model_id,
                        "trial": trial,
                        "feature_id": feature_id,
                        "feature_name": FEATURE_BY_ID[feature_id]["feature_name"],
                        "rank": rank,
                        "borda_score": (7 - rank) / 6,
                        "presentation_position": position_by_feature[feature_id],
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 840:
        raise AssertionError(f"rankings_long must have 840 rows, got {len(result)}")
    return result


def percentile_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    n = len(values)
    sample_indices = rng.integers(0, n, size=(replicates, n))
    bootstrap_means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def model_feature_statistics(long_df: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = iter(seed_sequence.spawn(len(MODEL_IDS) * len(FEATURE_IDS)))
    rows = []
    for model_id in MODEL_IDS:
        for feature_id in FEATURE_IDS:
            group = long_df[
                long_df.model_id.eq(model_id) & long_df.feature_id.eq(feature_id)
            ].sort_values("trial")
            if len(group) != 30:
                raise AssertionError(f"Expected 30 observations for {(model_id, feature_id)}, got {len(group)}")
            scores = group.borda_score.to_numpy(dtype=float)
            ranks = group["rank"].to_numpy(dtype=float)
            rng = np.random.default_rng(next(child_seeds))
            ci_lower, ci_upper = percentile_ci(scores, rng, replicates)
            rows.append(
                {
                    "model": MODEL_BY_ID[model_id]["model"],
                    "model_id": model_id,
                    "feature_id": feature_id,
                    "feature_name": FEATURE_BY_ID[feature_id]["feature_name"],
                    "mean_borda": float(scores.mean()),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "mean_rank": float(ranks.mean()),
                    "median_rank": float(np.median(ranks)),
                    "rank_sd": float(ranks.std(ddof=1)),
                    "n_trials": len(group),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 28:
        raise AssertionError(f"model_feature_importance must have 28 rows, got {len(result)}")
    return result


def trial_rank_vectors(long_df: pd.DataFrame, model_id: str) -> np.ndarray:
    pivot = (
        long_df[long_df.model_id.eq(model_id)]
        .pivot(index="trial", columns="feature_id", values="rank")
        .reindex(index=range(1, 31), columns=FEATURE_IDS)
    )
    if pivot.isna().any().any() or pivot.shape != (30, 7):
        raise AssertionError(f"Incomplete trial-rank matrix for {model_id}: {pivot.shape}")
    return pivot.to_numpy(dtype=float)


def tau_matrix_for_trials(rank_vectors: np.ndarray) -> np.ndarray:
    n = len(rank_vectors)
    matrix = np.eye(n, dtype=float)
    for i, j in itertools.combinations(range(n), 2):
        tau = float(kendalltau(rank_vectors[i], rank_vectors[j]).statistic)
        matrix[i, j] = matrix[j, i] = tau
    return matrix


def bootstrap_mean_pairwise_tau(
    tau_matrix: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    n = tau_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    means = np.empty(replicates, dtype=float)

    for start in range(0, replicates, 1_000):
        stop = min(start + 1_000, replicates)
        sampled = rng.integers(0, n, size=(stop - start, n))
        resampled_taus = tau_matrix[sampled[:, upper_i], sampled[:, upper_j]]
        means[start:stop] = resampled_taus.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def within_model_stability(
    long_df: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = iter(seed_sequence.spawn(len(MODEL_IDS)))
    summary_rows, pair_rows = [], []
    for model_id in MODEL_IDS:
        vectors = trial_rank_vectors(long_df, model_id)
        tau_matrix = tau_matrix_for_trials(vectors)
        pair_taus = []
        for trial_a, trial_b in itertools.combinations(range(30), 2):
            tau = float(tau_matrix[trial_a, trial_b])
            pair_taus.append(tau)
            pair_rows.append(
                {
                    "model": MODEL_BY_ID[model_id]["model"],
                    "model_id": model_id,
                    "trial_a": trial_a + 1,
                    "trial_b": trial_b + 1,
                    "kendall_tau": tau,
                }
            )
        values = np.asarray(pair_taus)
        if len(values) != 435:
            raise AssertionError(f"Expected 435 trial pairs for {model_id}, got {len(values)}")
        rng = np.random.default_rng(next(child_seeds))
        ci_lower, ci_upper = bootstrap_mean_pairwise_tau(tau_matrix, rng, replicates)
        summary_rows.append(
            {
                "model": MODEL_BY_ID[model_id]["model"],
                "model_id": model_id,
                "mean_kendall_tau": float(values.mean()),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "median_kendall_tau": float(np.median(values)),
                "kendall_tau_sd": float(values.std(ddof=1)),
                "min_kendall_tau": float(values.min()),
                "max_kendall_tau": float(values.max()),
                "n_trial_pairs": len(values),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def intermodel_agreement(importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rank_vectors = {}
    for model_id in MODEL_IDS:
        scores = (
            importance[importance.model_id.eq(model_id)]
            .set_index("feature_id")
            .reindex(FEATURE_IDS)
            .mean_borda.to_numpy(dtype=float)
        )
        # Average ranks preserve ties
        final_rank_vectors[model_id] = rankdata(-scores, method="average")

    matrix = pd.DataFrame(np.eye(len(MODEL_IDS)), index=MODEL_IDS, columns=MODEL_IDS, dtype=float)
    pair_rows = []
    for model_a, model_b in itertools.combinations(MODEL_IDS, 2):
        tau = float(kendalltau(final_rank_vectors[model_a], final_rank_vectors[model_b]).statistic)
        matrix.loc[model_a, model_b] = matrix.loc[model_b, model_a] = tau
        pair_rows.append(
            {
                "model_a": MODEL_BY_ID[model_a]["model"],
                "model_a_id": model_a,
                "model_b": MODEL_BY_ID[model_b]["model"],
                "model_b_id": model_b,
                "kendall_tau": tau,
            }
        )
    matrix.index = [MODEL_BY_ID[model_id]["model"] for model_id in matrix.index]
    matrix.columns = [MODEL_BY_ID[model_id]["model"] for model_id in matrix.columns]
    matrix.index.name = "model"
    return matrix, pd.DataFrame(pair_rows)


def main_results_table(importance: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature_id": FEATURE_IDS,
            "feature_name": [FEATURE_BY_ID[feature_id]["feature_name"] for feature_id in FEATURE_IDS],
        }
    )
    for model in MODELS:
        group = importance[importance.model_id.eq(model["model_id"])].set_index("feature_id").reindex(FEATURE_IDS)
        label = model["model"]
        table[f"{label} Borda"] = group.mean_borda.to_numpy()
        table[f"{label} CI lower"] = group.ci_lower.to_numpy()
        table[f"{label} CI upper"] = group.ci_upper.to_numpy()
        table[f"{label} 95% CI"] = [
            f"[{lower:.3f}, {upper:.3f}]" for lower, upper in zip(group.ci_lower, group.ci_upper)
        ]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")

    write_protocol_files(args.out)
    manifest_path = args.out / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition = manifest.get("condition", "no_data")
    raw_path = args.out / "raw_llm_responses.jsonl"
    records = load_records(raw_path)
    selected = select_complete_valid_records(records, manifest["protocol_sha256"], condition)

    long_df = build_rankings_long(selected)
    importance = model_feature_statistics(long_df, args.bootstrap_replicates, args.bootstrap_seed)
    stability, pairwise_stability = within_model_stability(
        long_df, args.bootstrap_replicates, args.bootstrap_seed
    )
    intermodel_matrix, intermodel_pairs = intermodel_agreement(importance)
    main_table = main_results_table(importance)

    long_df.to_csv(args.out / "rankings_long.csv", index=False)
    importance.to_csv(args.out / "model_feature_importance.csv", index=False)
    stability.to_csv(args.out / "model_stability_kendall.csv", index=False)
    pairwise_stability.to_csv(args.out / "model_stability_pairwise_kendall.csv", index=False)
    intermodel_matrix.to_csv(args.out / "intermodel_kendall.csv")
    intermodel_pairs.to_csv(args.out / "intermodel_kendall_pairs.csv", index=False)
    main_table.to_csv(args.out / "main_feature_importance_table.csv", index=False)

    print(f"Validated 120 rankings and wrote 840 feature-level rows to {args.out}")
    print("Model-feature rows: 28; within-model trial pairs: 4 x 435; inter-model pairs: 6")


if __name__ == "__main__":
    main()
