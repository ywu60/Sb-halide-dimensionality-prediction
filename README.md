# Sb-halide dimensionality prediction

Research code for binary prediction of inorganic Sb-halide connectivity:
`0 = 0D` and `1 = non-0D` (1D, 2D, or 3D).

## Project overview

This project studies the structural dimensionality of organic–inorganic antimony-halide compounds. We first
identified relevant compounds across a broad collection of chemistry papers and extracted their compositions,
organic cations, structural information, and experimentally reported dimensionalities. These literature data
form the labeled dataset used throughout the project.

We then trained two conventional machine-learning classifiers—Random Forest and support vector machine
(SVM)—to predict whether the inorganic Sb-halide connectivity is zero-dimensional (0D) or extended (non-0D,
including 1D, 2D, and 3D structures). To evaluate the capabilities of current general-purpose AI systems on the
same scientific task, we tested several GPT models using zero-shot, few-shot, and all-shot prompting and compared
their held-out performance with the conventional ML baselines.

To understand the basis of these predictions, we analyzed the trained ML models using permutation importance and separately asked the GPT models to rank a set of chemical and compositional features. We want to see how feature importance differs across these models.

Finally, we synthesized ten new Sb-halide compounds and applied the best-performing ML and GPT configurations to
predict their dimensionality. This provides an external application of the models beyond the literature-derived
training and test dataset.


## Layout

- `ml_training/`: feature engineering, leakage-controlled splitting, Random Forest, and SVM training.
- `model_prediction/`: feature generation and full-data ML prediction for new compounds.
- `feature_importance/`: permutation and SHAP importance for the selected ML model.
- `llm_prediction/`: held-out LLM evaluation and full-data in-context prediction for new compounds.
- `llm_feature_importance/`: final fixed-category paired experiment, with and without labeled examples.
- `evaluation/`: ML/LLM comparison and confusion-matrix plots.
- `common/`: shared plotting utilities.


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`scikit-learn==1.7.2` is pinned because it reproduces the published grouped-CV folds and predictions with
`random_state=42`; newer scikit-learn versions can produce different grouped folds despite using the same seed.

SMI-TED itself must also be available when creating embeddings. Supply its `smi_ted_light` directory with
`--smi-ted-dir`; the training-data preparation script can download the required files from Hugging Face if the
argument is omitted.

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

Set `OPENAI_API_KEY`, then run:

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

The final implementation uses seven fixed categories, four LLMs, 30 controlled presentation orders, and paired
conditions with and without the 321 training examples. 


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
