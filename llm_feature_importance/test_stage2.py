"""Local integrity tests for the Stage 2 protocol, validation, and statistics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from llm_feature_importance.stage2_analyze import (
    build_rankings_long,
    intermodel_agreement,
    model_feature_statistics,
    within_model_stability,
)
from llm_feature_importance.stage2_collect import JsonlLogger, collect_one, jsonable, parse_ranking
from llm_feature_importance.stage2_compare import (
    condition_kendall,
    paired_delta,
    plot_delta_heatmap,
    plot_stability_comparison,
    stability_comparison,
    validate_long,
)
from llm_feature_importance.stage2_collect_paired import collect_pair, paired_condition_order
from llm_feature_importance.stage2_config import (
    FEATURE_BY_ID,
    FEATURE_IDS,
    MODELS,
    MODEL_IDS,
    PRESENTATION_ORDERS,
    build_prompt,
    protocol_sha256,
)
from llm_feature_importance.stage2_dataset import DEFAULT_DATA_PATH, build_dataset_contexts, protocol_dataset_metadata
from llm_feature_importance.stage2_plot import (
    plot_importance_ci,
    plot_importance_heatmap,
    plot_intermodel_kendall,
    plot_stability,
)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DEFAULT_DATA_PATH.exists():
            cls.dataset_contexts, cls.dataset_examples = build_dataset_contexts()
        else:
            cls.dataset_contexts, cls.dataset_examples = None, None

    def test_no_data_prompt_is_unchanged_and_v2_protocol_is_locked(self):
        self.assertEqual(
            protocol_sha256(),
            "da905e55a5eac9c5186e8c9c97208ae92cb016a6da4ad04be0ebc8e5edbf21f8",
        )
        self.assertEqual(
            __import__("hashlib").sha256(build_prompt(PRESENTATION_ORDERS[0]).encode()).hexdigest(),
            "144034e6bb4f52e3170e2f705bc009104c1aab5cfd9444b6c6fd590d8a2b702f",
        )

    def test_permutations_are_fixed_and_complete(self):
        self.assertEqual(len(PRESENTATION_ORDERS), 30)
        self.assertEqual(len(set(PRESENTATION_ORDERS)), 30)
        for order in PRESENTATION_ORDERS:
            self.assertEqual(set(order), set(FEATURE_IDS))
            self.assertEqual(len(order), 7)

    def test_every_prompt_contains_each_definition_once(self):
        for order in PRESENTATION_ORDERS:
            prompt = build_prompt(order)
            for feature_id in FEATURE_IDS:
                feature = FEATURE_BY_ID[feature_id]
                self.assertEqual(prompt.count(feature["definition"]), 1)
                self.assertEqual(prompt.count(f"{feature_id}. {feature['feature_name']}"), 1)
            positions = [prompt.index(f"{feature_id}. {FEATURE_BY_ID[feature_id]['feature_name']}") for feature_id in order]
            self.assertEqual(positions, sorted(positions))

    def test_exact_old_321_examples_and_model_representations(self):
        if self.dataset_contexts is None:
            self.skipTest(f"private prepared dataset not present at {DEFAULT_DATA_PATH}")
        expected = {
            "gpt-5.1-2025-11-13": (
                "name", "82c5f89705ad6918bd67e89b6e816fb4da59862abb869a7f19d785d6364fad05"
            ),
            "gpt-5.5-2026-04-23": (
                "smiles", "743891999057f02ee722bda1547399aacf08917dd5b95a58f269da995153126d"
            ),
            "gpt-5.6-sol": (
                "smiles", "743891999057f02ee722bda1547399aacf08917dd5b95a58f269da995153126d"
            ),
            "gpt-6-astra": (
                "name", "82c5f89705ad6918bd67e89b6e816fb4da59862abb869a7f19d785d6364fad05"
            ),
        }
        self.assertEqual(len(self.dataset_examples), 642)
        self.assertEqual(
            self.dataset_examples.groupby("representation").size().to_dict(),
            {"name": 321, "smiles": 321},
        )
        for model_id, (representation, expected_hash) in expected.items():
            context = self.dataset_contexts[model_id]
            self.assertEqual(context.n_examples, 321)
            self.assertEqual(context.representation, representation)
            self.assertEqual(context.sha256, expected_hash)
            prompt = build_prompt(
                PRESENTATION_ORDERS[0], "with_data", context.block, context.n_examples
            )
            self.assertIn(context.block, prompt)
            self.assertEqual(prompt.count("Answer:"), 321)
            self.assertIn("0 = 0D and 1 = non-0D (including 1D, 2D, and 3D)", prompt)
            self.assertIn(
                "Use these examples to learn the relationships between the provided "
                "chemical/compositional information and structural dimensionality.",
                prompt,
            )
            for feature in FEATURE_BY_ID.values():
                self.assertEqual(prompt.count(feature["definition"]), 1)

    def test_with_data_protocol_records_per_model_dataset_hashes(self):
        if self.dataset_contexts is None:
            self.skipTest(f"private prepared dataset not present at {DEFAULT_DATA_PATH}")
        metadata = protocol_dataset_metadata(self.dataset_contexts)
        self.assertEqual(metadata["dataset_n_examples"], 321)
        self.assertEqual(set(metadata["contexts_by_model"]), set(MODEL_IDS))
        self.assertNotEqual(protocol_sha256("with_data", metadata), protocol_sha256())

    def test_both_conditions_use_the_same_trial_permutation(self):
        if self.dataset_contexts is None:
            self.skipTest(f"private prepared dataset not present at {DEFAULT_DATA_PATH}")
        for model_id in MODEL_IDS:
            context = self.dataset_contexts[model_id]
            for order in PRESENTATION_ORDERS:
                no_data = build_prompt(order)
                with_data = build_prompt(order, "with_data", context.block, 321)
                no_positions = [no_data.index(f"{feature_id}.") for feature_id in order]
                with_positions = [with_data.rindex(f"{feature_id}.") for feature_id in order]
                self.assertEqual(no_positions, sorted(no_positions))
                self.assertEqual(with_positions, sorted(with_positions))

    def test_paired_execution_order_is_counterbalanced(self):
        orders = [paired_condition_order(trial) for trial in range(1, 31)]
        self.assertEqual(sum(order[0] == "no_data" for order in orders), 15)
        self.assertEqual(sum(order[0] == "with_data" for order in orders), 15)
        self.assertEqual(orders[0], ("no_data", "with_data"))
        self.assertEqual(orders[1], ("with_data", "no_data"))

    def test_each_trial_executes_both_conditions_sequentially(self):
        observed = []

        def fake_collect_one(**kwargs):
            observed.append(
                (
                    kwargs["condition"],
                    kwargs["trial_id"],
                    kwargs["presentation_order"],
                    kwargs["paired_execution"],
                )
            )
            return list(FEATURE_IDS)

        model = MODELS[0]
        with patch("llm_feature_importance.stage2_collect_paired.collect_one", side_effect=fake_collect_one):
            successes, failures = collect_pair(
                client=SimpleNamespace(),
                loggers={"no_data": SimpleNamespace(), "with_data": SimpleNamespace()},
                model=model,
                trial_id=2,
                contexts={model["model_id"]: SimpleNamespace()},
                hashes={"no_data": "no-hash", "with_data": "with-hash"},
                complete={"no_data": set(), "with_data": set()},
                max_attempts=1,
            )
        self.assertFalse(failures)
        self.assertEqual([condition for condition, _ in successes], ["with_data", "no_data"])
        self.assertEqual([row[0] for row in observed], ["with_data", "no_data"])
        self.assertEqual(observed[0][2], PRESENTATION_ORDERS[1])
        self.assertEqual(observed[1][2], PRESENTATION_ORDERS[1])
        self.assertEqual(observed[0][3]["pair_position"], 1)
        self.assertEqual(observed[1][3]["pair_position"], 2)


class ResponseValidationTests(unittest.TestCase):
    def test_collector_passes_reasoning_effort_directly(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    id="test-response",
                    model=MODEL_IDS[0],
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps({"ranking": list(FEATURE_IDS)})
                            ),
                            finish_reason="stop",
                        )
                    ],
                )

        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = JsonlLogger(Path(temp_dir) / "raw.jsonl")
            collect_one(
                client,
                logger,
                MODEL_IDS[0],
                MODELS[0]["model"],
                1,
                PRESENTATION_ORDERS[0],
                protocol_sha256(),
                1,
            )
        self.assertEqual(completions.kwargs["reasoning_effort"], "medium")
        self.assertNotIn("extra_body", completions.kwargs)

    def test_usage_serialization_falls_back_for_legacy_sdk_objects(self):
        class LegacyUsage:
            prompt_tokens = 12

            def __init__(self):
                self.prompt_tokens = 12
                self.details = {"reasoning_tokens": 3}

            def model_dump(self):
                raise AttributeError("missing __pydantic_serializer__")

            def dict(self):
                raise AttributeError("missing __pydantic_serializer__")

        self.assertEqual(jsonable(LegacyUsage()), {"prompt_tokens": 12, "details": {"reasoning_tokens": 3}})

    def test_accepts_exact_permutation(self):
        raw = json.dumps({"ranking": list(FEATURE_IDS)})
        self.assertEqual(parse_ranking(raw), list(FEATURE_IDS))

    def test_rejects_free_text_or_invalid_ids(self):
        invalid = [
            "Here is my ranking: " + json.dumps({"ranking": list(FEATURE_IDS)}),
            json.dumps({"ranking": list(FEATURE_IDS[:-1])}),
            json.dumps({"ranking": ["F1", "F1", "F2", "F3", "F4", "F5", "F6"]}),
            json.dumps({"ranking": ["F1", "F2", "F3", "F4", "F5", "F6", "cation size"]}),
            json.dumps({"ranking": list(FEATURE_IDS), "explanation": "not allowed"}),
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_ranking(raw)


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def synthetic_selected():
        selected = {}
        protocol_hash = protocol_sha256()
        base = list(FEATURE_IDS)
        for model_index, model_id in enumerate(MODEL_IDS):
            for trial in range(1, 31):
                shift = (model_index + trial - 1) % len(base)
                ranking = base[shift:] + base[:shift]
                selected[(model_id, trial)] = {
                    "model_id": model_id,
                    "trial_id": trial,
                    "protocol_sha256": protocol_hash,
                    "presentation_order": list(PRESENTATION_ORDERS[trial - 1]),
                    "raw_response": json.dumps({"ranking": ranking}),
                    "parsed_ranking": ranking,
                    "valid": True,
                }
        return selected

    def test_all_analysis_shapes_and_invariants(self):
        long_df = build_rankings_long(self.synthetic_selected())
        self.assertEqual(len(long_df), 840)
        self.assertEqual(long_df.groupby(["model_id", "trial"]).size().unique().tolist(), [7])
        self.assertTrue(np.allclose(long_df.borda_score, (7 - long_df["rank"]) / 6))
        self.assertTrue((long_df.groupby(["model_id", "trial"]).borda_score.sum() == 3.5).all())

        importance = model_feature_statistics(long_df, replicates=100, seed=20260909)
        self.assertEqual(len(importance), 28)
        self.assertTrue((importance.ci_lower <= importance.mean_borda).all())
        self.assertTrue((importance.mean_borda <= importance.ci_upper).all())

        stability, pairs = within_model_stability(long_df, replicates=100, seed=20260909)
        self.assertEqual(len(stability), 4)
        self.assertEqual(len(pairs), 4 * 435)
        self.assertFalse(stability.isna().any().any())

        matrix, model_pairs = intermodel_agreement(importance)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(len(model_pairs), 6)
        self.assertTrue(np.allclose(matrix.to_numpy(), matrix.to_numpy().T))
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            plot_importance_heatmap(importance, output)
            plot_importance_ci(importance, output)
            plot_stability(stability, output)
            plot_intermodel_kendall(matrix.reset_index(), output)
            self.assertEqual(len(list(output.glob("*.png"))), 4)
            self.assertEqual(len(list(output.glob("*.pdf"))), 4)

    def test_paired_condition_comparison(self):
        no_data = validate_long(build_rankings_long(self.synthetic_selected()), "no_data")
        with_data = no_data.copy()
        with_data["rank"] = 8 - with_data["rank"]
        with_data["borda_score"] = (7 - with_data["rank"]) / 6
        with_data = validate_long(with_data, "with_data")

        delta, paired = paired_delta(no_data, with_data, replicates=100, seed=20260909)
        self.assertEqual(len(paired), 840)
        self.assertEqual(len(delta), 28)
        self.assertTrue((delta.n_paired_trials == 30).all())
        self.assertTrue(
            np.allclose(
                paired.delta_borda,
                paired.borda_score_with_data - paired.borda_score_no_data,
            )
        )
        agreement = condition_kendall(no_data, with_data)
        self.assertEqual(len(agreement), 4)
        self.assertFalse(agreement.isna().any().any())

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            no_dir, with_dir, output = root / "no", root / "with", root / "comparison"
            no_dir.mkdir()
            with_dir.mkdir()
            output.mkdir()
            base_stability = pd.DataFrame(
                {
                    "model": [model["model"] for model in MODELS],
                    "model_id": list(MODEL_IDS),
                    "mean_kendall_tau": [0.1, 0.2, 0.3, 0.4],
                    "ci_lower": [0.0, 0.1, 0.2, 0.3],
                    "ci_upper": [0.2, 0.3, 0.4, 0.5],
                }
            )
            base_stability.to_csv(no_dir / "model_stability_kendall.csv", index=False)
            changed_stability = base_stability.copy()
            changed_stability[["mean_kendall_tau", "ci_lower", "ci_upper"]] += 0.1
            changed_stability.to_csv(with_dir / "model_stability_kendall.csv", index=False)
            stability = stability_comparison(no_dir, with_dir)
            self.assertTrue(np.allclose(stability.delta_mean_kendall_tau, 0.1))
            plot_delta_heatmap(delta, output)
            plot_stability_comparison(stability, output)
            self.assertEqual(len(list(output.glob("*.png"))), 2)
            self.assertEqual(len(list(output.glob("*.pdf"))), 2)


if __name__ == "__main__":
    unittest.main()
