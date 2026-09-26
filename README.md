# Sb-halide dimensionality prediction

Research code for binary prediction of inorganic Sb-halide connectivity:
`0 = 0D` and `1 = non-0D` (1D, 2D, or 3D).

## Project overview

This project studies the structural dimensionality of organic–inorganic
antimony-halide compounds. We first identify relevant compounds in chemistry
papers and extract their compositions, organic cations, structural information,
and reported dimensionalities. These literature data form the labeled dataset
used throughout the project.

We train Random Forest and support vector machine (SVM) classifiers to predict
whether inorganic Sb-halide connectivity is zero-dimensional or extended. We
also evaluate GPT models on the same held-out task using zero-shot, few-shot,
and all-shot prompting.

To understand the predictions, we calculate permutation and SHAP importance for
the ML models and ask the GPT models to rank the same chemical and compositional
features. Finally, we apply the best-performing ML and GPT configurations to ten
newly synthesized Sb-halide compounds.

## Repository layout

The directories follow the research workflow from literature extraction to
model evaluation and prediction:

- `data_extraction/`: PDF-to-dataset pipeline with evidence-linked Excel and JSON export.
- `data/`: prepared modeling datasets used by the downstream workflows.
- `ml_training/`: feature engineering, leakage-controlled splitting, Random Forest, and SVM training.
- `feature_importance/`: permutation and SHAP importance for the selected ML model.
- `llm_prediction/`: held-out LLM evaluation and in-context prediction for new compounds.
- `llm_feature_importance/`: paired feature-ranking experiments with and without labeled examples.
- `model_prediction/`: feature generation and full-data ML prediction for new compounds.
- `evaluation/`: ML/LLM comparisons and confusion-matrix plots.
- `common/`: shared plotting utilities.
- `results_ml/`: generated ML metrics, predictions, and fitted models.
- `figures/`: generated publication and diagnostic figures.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`scikit-learn==1.7.2` is pinned because it reproduces the published grouped-CV
folds and predictions with `random_state=42`; newer versions can produce
different grouped folds despite using the same seed.

SMI-TED must be available when creating molecular embeddings. Supply its
`smi_ted_light` directory with `--smi-ted-dir`; the preparation script can
download the required files from Hugging Face if the argument is omitted.

Set `OPENAI_API_KEY` before live data-extraction or LLM-prediction runs.

## Literature data extraction

The `data_extraction/` package builds the literature-derived dataset consumed by
the prediction workflows. It combines deterministic PDF parsing, BM25 retrieval,
validation, source citation tracking, and final export with schema-constrained LLM
stages for scientific interpretation.

```text
PDFs -> parse -> screen -> compound registry -> retrieve -> dossiers
     -> structured extraction -> verification -> final Excel + JSON datasets
```

Each extracted compound retains its supporting evidence text and page-level
source IDs. The pipeline produces one record per compound, including reported
formula and cation information, Sb-halide connectivity, dimensionality and
reasoning, synthesis evidence, verification results, and automatic flags.

From the repository root, run the extraction pipeline for a selected set of papers:

```bash
cd data_extraction
python -m src.run_pipeline all --paper-ids P0001,P0002
```

The final stage writes both formats directly:

```text
data/output/final_dataset.xlsx
data/output/final_dataset.json
```

## ML workflow

```bash
python -m ml_training.prepare_data \
  --input data/merged_compound_cif_cation.xlsx \
  --output data/prepared_data.xlsx

python -m ml_training.train_models \
  --data data/prepared_data.xlsx \
  --out results_ml

python -m feature_importance.permutation_importance \
  --data data/prepared_data.xlsx \
  --results results_ml/ml_results.csv

python -m feature_importance.shap_importance \
  --data data/prepared_data.xlsx \
  --results results_ml/ml_results.csv
```

For the project-specific mixed-halide candidates:

```bash
python -m model_prediction.featurize_new_compounds \
  --input data/new_mixed_halide_sb_compounds.xlsx \
  --smi-ted-dir /path/to/smi_ted_light \
  --output data/new_compound_features.csv

python -m model_prediction.retrain_and_predict \
  --data data/prepared_data.xlsx \
  --new-features data/new_compound_features.csv \
  --out results_ml/new_compounds
```

## LLM prediction

```bash
python -m llm_prediction.evaluate_llms \
  --data data/prepared_data.xlsx \
  --out results_llm

python -m llm_prediction.predict_new_compounds \
  --data data/prepared_data.xlsx \
  --new-features data/new_compound_features.csv \
  --output results_llm/new_compound_predictions.csv
```

## LLM feature importance

The final experiment uses seven fixed feature categories, four LLMs, 30
controlled presentation orders, and paired conditions with and without the 321
training examples.

```bash
python -m unittest -v llm_feature_importance.test_stage2

python -m llm_feature_importance.stage2_collect_paired \
  --data data/prepared_data.xlsx

python -m llm_feature_importance.stage2_analyze --out llm_feature_importance/results
python -m llm_feature_importance.stage2_analyze --out llm_feature_importance/results_with_data
python -m llm_feature_importance.stage2_plot --out llm_feature_importance/results
python -m llm_feature_importance.stage2_plot --out llm_feature_importance/results_with_data
python -m llm_feature_importance.stage2_compare \
  --no-data llm_feature_importance/results \
  --with-data llm_feature_importance/results_with_data \
  --out llm_feature_importance/comparison
```
