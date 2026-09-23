"""Compare paired no-data and with-data Stage 2 feature-importance results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from llm_feature_importance.stage2_analyze import percentile_ci
from llm_feature_importance.stage2_config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FEATURE_BY_ID,
    FEATURE_IDS,
    FIGURE_FEATURE_LABELS,
    MODELS,
    MODEL_BY_ID,
    MODEL_IDS,
    MODEL_NAMES,
    PRESENTATION_ORDERS,
)


BG, INK, GRID = "#ffffff", "#111111", "#dedede"
NO_DATA_COLOR, WITH_DATA_COLOR = "#7f8c8d", "#2a78d6"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.linewidth": 0.8,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")


def load_manifest(directory: Path, expected_condition: str) -> dict:
    path = directory / "run_manifest.json"
    if not path.exists():
        raise RuntimeError(f"Missing manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    condition = manifest.get("condition", "no_data")
    if condition != expected_condition:
        raise RuntimeError(f"Expected {expected_condition} at {directory}, manifest says {condition}")
    return manifest


def validate_long(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "model", "model_id", "trial", "feature_id", "feature_name",
        "rank", "borda_score", "presentation_position",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{label} rankings_long.csv lacks columns: {sorted(missing)}")
    if len(df) != 840:
        raise RuntimeError(f"{label} must contain 840 rows, found {len(df)}")
    key = ["model_id", "trial", "feature_id"]
    if df.duplicated(key).any():
        raise RuntimeError(f"{label} contains duplicate paired keys")
    if set(df.model_id) != set(MODEL_IDS) or set(df.feature_id) != set(FEATURE_IDS):
        raise RuntimeError(f"{label} does not contain exactly the four models and F1-F7")
    counts = df.groupby("model_id").trial.nunique()
    if not counts.reindex(MODEL_IDS).eq(30).all():
        raise RuntimeError(f"{label} does not have 30 trials per model: {counts.to_dict()}")
    if not np.allclose(df.borda_score, (7 - df["rank"]) / 6):
        raise RuntimeError(f"{label} has a Borda score inconsistent with rank")
    for model_id in MODEL_IDS:
        for trial, order in enumerate(PRESENTATION_ORDERS, start=1):
            group = df[df.model_id.eq(model_id) & df.trial.eq(trial)]
            observed = dict(zip(group.feature_id, group.presentation_position))
            expected = {feature_id: position for position, feature_id in enumerate(order, start=1)}
            if observed != expected:
                raise RuntimeError(f"{label} presentation order mismatch for {(model_id, trial)}")
    return df.sort_values(key).reset_index(drop=True)


def paired_delta(
    no_data: pd.DataFrame,
    with_data: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["model_id", "trial", "feature_id"]
    paired = no_data.merge(
        with_data,
        on=key,
        suffixes=("_no_data", "_with_data"),
        validate="one_to_one",
    )
    if len(paired) != 840:
        raise RuntimeError(f"Expected 840 paired feature observations, found {len(paired)}")
    if not (paired.presentation_position_no_data == paired.presentation_position_with_data).all():
        raise RuntimeError("The two conditions do not use the same paired trial presentation positions")
    paired["delta_borda"] = paired.borda_score_with_data - paired.borda_score_no_data

    seeds = iter(np.random.SeedSequence(seed).spawn(len(MODEL_IDS) * len(FEATURE_IDS)))
    rows = []
    for model_id in MODEL_IDS:
        for feature_id in FEATURE_IDS:
            group = paired[paired.model_id.eq(model_id) & paired.feature_id.eq(feature_id)].sort_values("trial")
            if len(group) != 30:
                raise RuntimeError(f"Expected 30 paired differences for {(model_id, feature_id)}")
            differences = group.delta_borda.to_numpy(dtype=float)
            lower, upper = percentile_ci(differences, np.random.default_rng(next(seeds)), replicates)
            rows.append(
                {
                    "model": MODEL_BY_ID[model_id]["model"],
                    "model_id": model_id,
                    "feature_id": feature_id,
                    "feature_name": FEATURE_BY_ID[feature_id]["feature_name"],
                    "no_data_mean_borda": float(group.borda_score_no_data.mean()),
                    "with_data_mean_borda": float(group.borda_score_with_data.mean()),
                    "delta_borda": float(differences.mean()),
                    "delta_ci_lower": lower,
                    "delta_ci_upper": upper,
                    "n_paired_trials": len(group),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 28:
        raise AssertionError(f"Expected 28 delta rows, found {len(result)}")
    return result, paired


def condition_kendall(no_data: pd.DataFrame, with_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        no_scores = (
            no_data[no_data.model_id.eq(model_id)]
            .groupby("feature_id").borda_score.mean().reindex(FEATURE_IDS).to_numpy()
        )
        with_scores = (
            with_data[with_data.model_id.eq(model_id)]
            .groupby("feature_id").borda_score.mean().reindex(FEATURE_IDS).to_numpy()
        )
        no_ranks = rankdata(-no_scores, method="average")
        with_ranks = rankdata(-with_scores, method="average")
        rows.append(
            {
                "model": MODEL_BY_ID[model_id]["model"],
                "model_id": model_id,
                "kendall_tau_no_data_vs_with_data": float(kendalltau(no_ranks, with_ranks).statistic),
            }
        )
    return pd.DataFrame(rows)


def stability_comparison(no_dir: Path, with_dir: Path) -> pd.DataFrame:
    no_data = pd.read_csv(no_dir / "model_stability_kendall.csv")
    with_data = pd.read_csv(with_dir / "model_stability_kendall.csv")
    required = {"model", "model_id", "mean_kendall_tau", "ci_lower", "ci_upper"}
    for label, frame in [("no_data", no_data), ("with_data", with_data)]:
        if required - set(frame):
            raise RuntimeError(f"{label} stability file is missing required columns")
        if len(frame) != 4 or set(frame.model_id) != set(MODEL_IDS):
            raise RuntimeError(f"{label} stability file must contain exactly the four models")
    merged = no_data[list(required)].merge(
        with_data[list(required)], on=["model", "model_id"], suffixes=("_no_data", "_with_data"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "model": merged.model,
            "model_id": merged.model_id,
            "no_data_mean_kendall_tau": merged.mean_kendall_tau_no_data,
            "no_data_ci_lower": merged.ci_lower_no_data,
            "no_data_ci_upper": merged.ci_upper_no_data,
            "with_data_mean_kendall_tau": merged.mean_kendall_tau_with_data,
            "with_data_ci_lower": merged.ci_lower_with_data,
            "with_data_ci_upper": merged.ci_upper_with_data,
            "delta_mean_kendall_tau": merged.mean_kendall_tau_with_data - merged.mean_kendall_tau_no_data,
        }
    ).set_index("model_id").reindex(MODEL_IDS).reset_index()


def plot_delta_heatmap(delta: pd.DataFrame, output: Path) -> None:
    pivot = delta.pivot(index="feature_id", columns="model", values="delta_borda").reindex(
        index=FEATURE_IDS, columns=MODEL_NAMES
    )
    values = pivot.to_numpy(dtype=float)
    limit = float(np.max(np.abs(values)))
    if limit == 0:
        limit = 1.0
    labels = [FIGURE_FEATURE_LABELS[feature_id] for feature_id in FEATURE_IDS]
    model_labels = [
        name.replace("GPT-5.6 Sol", "GPT-5.6\nSol").replace("GPT-6 Astra", "GPT-6\nAstra")
        for name in MODEL_NAMES
    ]
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(FEATURE_IDS)):
        for j in range(len(MODEL_NAMES)):
            value = values[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=14,
                    color=BG if abs(value) > 0.62 * limit else INK)
    ax.set_xticks(range(len(MODEL_NAMES)), model_labels, fontsize=13)
    ax.set_yticks(range(len(FEATURE_IDS)), labels, fontsize=18)
    ax.set_xticks(np.arange(-0.5, len(MODEL_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FEATURE_IDS), 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    colorbar.set_label("Change in mean Borda (with data − no data)", fontsize=17)
    colorbar.ax.tick_params(labelsize=15)
    save(fig, output / "comparison_delta_heatmap")


def plot_stability_comparison(stability: pd.DataFrame, output: Path) -> None:
    stability = stability.set_index("model_id").reindex(MODEL_IDS)
    y = np.arange(len(MODEL_IDS), dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for offset, prefix, label, color in [
        (-0.10, "no_data", "Without data", NO_DATA_COLOR),
        (0.10, "with_data", "With data", WITH_DATA_COLOR),
    ]:
        means = stability[f"{prefix}_mean_kendall_tau"].to_numpy(dtype=float)
        lower = stability[f"{prefix}_ci_lower"].to_numpy(dtype=float)
        upper = stability[f"{prefix}_ci_upper"].to_numpy(dtype=float)
        yy = y + offset
        # Draw interval endpoints directly. Percentile intervals can very rarely
        # exclude the observed statistic, which matplotlib.errorbar rejects as a
        # negative error length even though the interval itself is valid.
        ax.hlines(yy, lower, upper, color=color, linewidth=1.4, zorder=2)
        ax.vlines(lower, yy - 0.035, yy + 0.035, color=color, linewidth=1.2, zorder=2)
        ax.vlines(upper, yy - 0.035, yy + 0.035, color=color, linewidth=1.2, zorder=2)
        ax.scatter(means, yy, s=48, color=color, label=label, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_yticks(y, MODEL_NAMES, fontsize=10.5)
    ax.invert_yaxis()
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Mean pairwise Kendall's τ (95% bootstrap CI)", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    save(fig, output / "comparison_stability")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-data", type=Path, required=True)
    parser.add_argument("--with-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")

    no_manifest = load_manifest(args.no_data, "no_data")
    with_manifest = load_manifest(args.with_data, "with_data")
    no_data = validate_long(pd.read_csv(args.no_data / "rankings_long.csv"), "no_data")
    with_data = validate_long(pd.read_csv(args.with_data / "rankings_long.csv"), "with_data")

    args.out.mkdir(parents=True, exist_ok=True)
    delta, paired = paired_delta(no_data, with_data, args.bootstrap_replicates, args.bootstrap_seed)
    condition_tau = condition_kendall(no_data, with_data)
    stability = stability_comparison(args.no_data, args.with_data)
    delta.to_csv(args.out / "comparison_delta_feature_importance.csv", index=False)
    condition_tau.to_csv(args.out / "comparison_condition_kendall.csv", index=False)
    stability.to_csv(args.out / "comparison_stability.csv", index=False)
    plot_delta_heatmap(delta, args.out)
    plot_stability_comparison(stability, args.out)
    comparison_manifest = {
        "no_data_protocol_sha256": no_manifest["protocol_sha256"],
        "with_data_protocol_sha256": with_manifest["protocol_sha256"],
        "n_paired_feature_rows": len(paired),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (args.out / "comparison_manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated and compared 840 paired feature observations; wrote results to {args.out}")


if __name__ == "__main__":
    main()
