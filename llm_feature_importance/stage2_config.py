"""Immutable protocol definition for the Stage 2 fixed-category experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_WITH_DATA_OUTPUT_DIR = ROOT / "results_with_data"
CONDITIONS = ("no_data", "with_data")

FEATURES = (
    {
        "feature_id": "F1",
        "feature_name": "Halide:Sb ratio",
        "definition": "Stoichiometric ratio of inorganic halide atoms (F, Cl, Br, or I) to Sb atoms in the compound.",
    },
    {
        "feature_id": "F2",
        "feature_name": "Sb oxidation state",
        "definition": "Formal oxidation state of Sb, including Sb(III), Sb(V), or mixed-valence cases.",
    },
    {
        "feature_id": "F3",
        "feature_name": "Halide composition",
        "definition": "Identity and relative composition of the inorganic halides (F, Cl, Br, I), including single- and mixed-halide systems.",
    },
    {
        "feature_id": "F4",
        "feature_name": "Organic cation molecular structure",
        "definition": "Molecular identity, connectivity, scaffold, shape, rigidity, and related structural characteristics of the organic cation, excluding its overall size and formal charge.",
    },
    {
        "feature_id": "F5",
        "feature_name": "Organic cation size / steric bulk",
        "definition": "Overall molecular size and steric bulk of the organic cation, excluding its formal charge.",
    },
    {
        "feature_id": "F6",
        "feature_name": "Organic cation charge / protonation state",
        "definition": "Formal charge and protonation state of the organic cation.",
    },
    {
        "feature_id": "F7",
        "feature_name": "Water content",
        "definition": "Presence and stoichiometric amount of water associated with the compound.",
    },
)

FEATURE_IDS = tuple(feature["feature_id"] for feature in FEATURES)
FEATURE_BY_ID = {feature["feature_id"]: feature for feature in FEATURES}

# Concise display-only labels for figures. The canonical feature names and
# definitions above remain unchanged in prompts, analyses, and CSV tables.
FIGURE_FEATURE_LABELS = {
    "F1": "Halide:Sb ratio",
    "F2": "Sb oxidation state",
    "F3": "Halide composition",
    "F4": "Organic-cation structure",
    "F5": "Cation size / steric bulk",
    "F6": "Cation charge / protonation",
    "F7": "Water content",
}

# The order for trial t is shared by all models. Trial IDs are one-based.
PRESENTATION_ORDERS = (
    ("F4", "F6", "F2", "F3", "F7", "F1", "F5"),
    ("F1", "F5", "F3", "F2", "F6", "F7", "F4"),
    ("F5", "F4", "F6", "F7", "F3", "F2", "F1"),
    ("F3", "F4", "F7", "F2", "F5", "F1", "F6"),
    ("F6", "F3", "F2", "F7", "F5", "F1", "F4"),
    ("F2", "F3", "F4", "F6", "F5", "F7", "F1"),
    ("F4", "F3", "F1", "F5", "F7", "F6", "F2"),
    ("F3", "F2", "F5", "F6", "F4", "F7", "F1"),
    ("F7", "F1", "F4", "F3", "F6", "F5", "F2"),
    ("F7", "F3", "F2", "F6", "F5", "F1", "F4"),
    ("F5", "F6", "F2", "F4", "F1", "F3", "F7"),
    ("F1", "F2", "F7", "F5", "F4", "F3", "F6"),
    ("F7", "F4", "F6", "F1", "F3", "F2", "F5"),
    ("F6", "F4", "F5", "F2", "F7", "F3", "F1"),
    ("F5", "F2", "F1", "F7", "F3", "F4", "F6"),
    ("F4", "F2", "F7", "F1", "F6", "F5", "F3"),
    ("F5", "F2", "F6", "F4", "F1", "F3", "F7"),
    ("F7", "F5", "F3", "F1", "F2", "F6", "F4"),
    ("F4", "F5", "F1", "F7", "F6", "F2", "F3"),
    ("F4", "F7", "F6", "F2", "F1", "F5", "F3"),
    ("F2", "F3", "F5", "F1", "F4", "F7", "F6"),
    ("F2", "F6", "F4", "F3", "F1", "F7", "F5"),
    ("F3", "F1", "F6", "F4", "F2", "F5", "F7"),
    ("F1", "F6", "F5", "F7", "F2", "F4", "F3"),
    ("F3", "F7", "F1", "F6", "F2", "F4", "F5"),
    ("F5", "F1", "F4", "F2", "F3", "F6", "F7"),
    ("F2", "F7", "F3", "F4", "F6", "F1", "F5"),
    ("F6", "F5", "F1", "F3", "F7", "F4", "F2"),
    ("F6", "F1", "F7", "F5", "F3", "F2", "F4"),
    ("F1", "F7", "F3", "F5", "F4", "F6", "F2"),
)

MODELS = (
    {"model_id": "gpt-5.1-2025-11-13", "model": "GPT-5.1"},
    {"model_id": "gpt-5.5-2026-04-23", "model": "GPT-5.5"},
    {"model_id": "gpt-5.6-sol", "model": "GPT-5.6 Sol"},
    {"model_id": "gpt-6-astra", "model": "GPT-6 Astra"},
)
MODEL_IDS = tuple(model["model_id"] for model in MODELS)
MODEL_BY_ID = {model["model_id"]: model for model in MODELS}
MODEL_NAMES = tuple(model["model"] for model in MODELS)

REASONING_EFFORT = "medium"
MAX_COMPLETION_TOKENS = 8192
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_909
PROTOCOL_VERSION = "stage2-fixed-category-v2"

PROMPT_PREFIX = """You are evaluating the importance of chemical and compositional features for predicting the structural dimensionality of Sb-halide compounds.

Below are seven predefined features. Rank all seven features according to their expected importance for predicting structural dimensionality, where rank 1 is the most important and rank 7 is the least important.

Use the definitions exactly as provided. Treat each feature as a separate factor. Do not combine features, introduce new features, modify the feature definitions, or omit any feature.

"""

PROMPT_SUFFIX = """

Return all seven feature IDs ordered from most important to least important. Each feature ID must appear exactly once.

Return JSON only, using exactly this object shape:
{"ranking":["<rank-1 feature ID>","<rank-2 feature ID>","<rank-3 feature ID>","<rank-4 feature ID>","<rank-5 feature ID>","<rank-6 feature ID>","<rank-7 feature ID>"]}"""

WITH_DATA_PREFIX = """You are evaluating the importance of chemical and compositional features for predicting the structural dimensionality of Sb-halide compounds.

Below are 321 labeled Sb-halide examples. The dimensionality label is encoded as 0 = 0D and 1 = non-0D (including 1D, 2D, and 3D). Use these examples to learn the relationships between the provided chemical/compositional information and structural dimensionality. Based on the patterns you infer from these examples, rank the seven predefined features according to their importance for dimensionality prediction.

"""

WITH_DATA_AFTER_EXAMPLES = """

Below are the seven predefined features and their definitions. Rank 1 is the most important and rank 7 is the least important.

Use the definitions exactly as provided. Treat each feature as a separate factor. Do not combine features, introduce new features, modify the feature definitions, or omit any feature.

"""


def validate_protocol() -> None:
    """Fail fast if a protocol constant was edited inconsistently."""
    expected = set(FEATURE_IDS)
    if FEATURE_IDS != tuple(f"F{i}" for i in range(1, 8)):
        raise ValueError(f"Feature IDs must be F1-F7 in order, got {FEATURE_IDS}")
    if len(PRESENTATION_ORDERS) != 30:
        raise ValueError(f"Expected 30 presentation orders, got {len(PRESENTATION_ORDERS)}")
    if len(set(PRESENTATION_ORDERS)) != 30:
        raise ValueError("Presentation orders must be unique")
    for trial, order in enumerate(PRESENTATION_ORDERS, start=1):
        if len(order) != 7 or set(order) != expected:
            raise ValueError(f"Trial {trial} is not a permutation of F1-F7: {order}")
    if len(MODEL_IDS) != 4 or len(set(MODEL_IDS)) != 4:
        raise ValueError("Exactly four unique model IDs are required")


def build_prompt(
    order: Iterable[str],
    condition: str = "no_data",
    dataset_block: str | None = None,
    dataset_n_examples: int = 0,
) -> str:
    """Build a trial prompt; only the order of these immutable blocks varies."""
    order = tuple(order)
    if len(order) != 7 or set(order) != set(FEATURE_IDS):
        raise ValueError(f"Not a valid feature permutation: {order}")
    blocks = [
        f"{feature_id}. {FEATURE_BY_ID[feature_id]['feature_name']}\n"
        f"Definition: {FEATURE_BY_ID[feature_id]['definition']}"
        for feature_id in order
    ]
    feature_text = "\n\n".join(blocks)
    if condition == "no_data":
        if dataset_block is not None or dataset_n_examples != 0:
            raise ValueError("no_data prompt cannot include a dataset block")
        # Preserve the original Stage 2 no-data prompt byte for byte.
        return PROMPT_PREFIX + feature_text + PROMPT_SUFFIX
    if condition == "with_data":
        if dataset_n_examples != 321 or not dataset_block:
            raise ValueError("with_data prompt requires the fixed 321-example dataset block")
        return WITH_DATA_PREFIX + dataset_block + WITH_DATA_AFTER_EXAMPLES + feature_text + PROMPT_SUFFIX
    raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")


def protocol_payload(condition: str = "no_data", dataset_metadata: dict | None = None) -> dict:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "features": FEATURES,
        "presentation_orders": PRESENTATION_ORDERS,
        "models": MODELS,
        "inference_settings": {
            "reasoning_effort": REASONING_EFFORT,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "response_format": {"type": "json_object"},
        },
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_suffix": PROMPT_SUFFIX,
    }
    if condition == "no_data":
        if dataset_metadata is not None:
            raise ValueError("no_data protocol cannot include dataset metadata")
        # Keep the legacy payload unchanged so all completed no-data checkpoints retain
        # their original protocol hash.
        return payload
    if condition == "with_data":
        if not dataset_metadata:
            raise ValueError("with_data protocol requires dataset metadata")
        return payload | {
            "condition": "with_data",
            "dataset_metadata": dataset_metadata,
            "with_data_prefix": WITH_DATA_PREFIX,
            "with_data_after_examples": WITH_DATA_AFTER_EXAMPLES,
        }
    raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")


def protocol_sha256(condition: str = "no_data", dataset_metadata: dict | None = None) -> str:
    encoded = json.dumps(
        protocol_payload(condition=condition, dataset_metadata=dataset_metadata),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_protocol_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "feature_definitions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_id", "feature_name", "definition"])
        writer.writeheader()
        writer.writerows(FEATURES)
    with (output_dir / "feature_permutations.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["trial"] + [f"position_{i}" for i in range(1, 8)] + ["presentation_order"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial, order in enumerate(PRESENTATION_ORDERS, start=1):
            row = {"trial": trial, **{f"position_{i}": value for i, value in enumerate(order, 1)}}
            row["presentation_order"] = ",".join(order)
            writer.writerow(row)


validate_protocol()
