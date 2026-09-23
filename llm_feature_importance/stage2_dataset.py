"""Recover the exact 321-example contexts used by the grounded experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_feature_importance.stage2_config import MODEL_IDS, ROOT
from llm_prediction import evaluate_llms as run_llm


REPOSITORY_ROOT = ROOT.parent
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "prepared_data.xlsx"
ORDERING_SEED = 42
BEST_REPRESENTATION = {
    "gpt-5.1-2025-11-13": "name",
    "gpt-5.5-2026-04-23": "smiles",
    "gpt-5.6-sol": "smiles",
    "gpt-6-astra": "name",
}


@dataclass(frozen=True)
class DatasetContext:
    model_id: str
    representation: str
    block: str
    n_examples: int
    sha256: str
    source: str
    source_sha256: str

    def call_metadata(self) -> dict:
        return {
            "dataset_n_examples": self.n_examples,
            "dataset_sha256": self.sha256,
            "dataset_source": self.source,
            "dataset_source_sha256": self.source_sha256,
            "dataset_representation": self.representation,
            "dataset_ordering_seed": ORDERING_SEED,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_contexts(data_path: Path = DEFAULT_DATA_PATH) -> tuple[dict[str, DatasetContext], pd.DataFrame]:
    """Build each model's block once with the exact rows, order, format, and representation."""
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Original prepared dataset not found: {data_path}")
    missing_models = set(MODEL_IDS) - set(BEST_REPRESENTATION)
    if missing_models:
        raise RuntimeError(f"BEST_REPRESENTATION lacks current models: {sorted(missing_models)}")

    frame = pd.read_excel(data_path)
    train = frame[frame.split.eq("train")].copy()
    if len(train) != 321:
        raise RuntimeError(f"Expected exactly 321 old training examples, found {len(train)} in {data_path}")
    if train.compound_id.nunique() != 321:
        raise RuntimeError("The old 321-row training set does not have 321 unique compound IDs")
    shots = run_llm.order_all_shots(train, ORDERING_SEED, None)
    if len(shots) != 321 or set(shots.index) != set(train.index):
        raise RuntimeError("Old ordering function did not return exactly the original 321 training rows")

    source_hash = file_sha256(data_path)
    block_by_representation = {}
    example_rows = []
    for representation in sorted({BEST_REPRESENTATION[model_id] for model_id in MODEL_IDS}):
        serialized_examples = []
        for position, (row_index, row) in enumerate(shots.iterrows(), start=1):
            example = run_llm.chemical_block(row, representation) + f"\nAnswer: {int(row.target_non0D)}"
            serialized_examples.append(example)
            example_rows.append(
                {
                    "representation": representation,
                    "example_position": position,
                    "source_row_index": row_index,
                    "compound_id": row.compound_id,
                    "target_non0D": int(row.target_non0D),
                    "example_sha256": hashlib.sha256(example.encode()).hexdigest(),
                }
            )
        block_by_representation[representation] = "\n\n".join(serialized_examples)

    contexts = {}
    for model_id in MODEL_IDS:
        representation = BEST_REPRESENTATION[model_id]
        block = block_by_representation[representation]
        contexts[model_id] = DatasetContext(
            model_id=model_id,
            representation=representation,
            block=block,
            n_examples=321,
            sha256=hashlib.sha256(block.encode()).hexdigest(),
            source=str(data_path),
            source_sha256=source_hash,
        )
    return contexts, pd.DataFrame(example_rows)


def protocol_dataset_metadata(contexts: dict[str, DatasetContext]) -> dict:
    """Compact deterministic metadata included in the with-data protocol hash."""
    return {
        "dataset_n_examples": 321,
        "dataset_source": next(iter(contexts.values())).source,
        "dataset_source_sha256": next(iter(contexts.values())).source_sha256,
        "dataset_ordering_seed": ORDERING_SEED,
        "contexts_by_model": {
            model_id: {
                "representation": contexts[model_id].representation,
                "dataset_sha256": contexts[model_id].sha256,
            }
            for model_id in MODEL_IDS
        },
    }
