
# Sb-halide dimensionality prediction

Research code for binary prediction of inorganic Sb-halide connectivity:
`0 = 0D` and `1 = non-0D` (1D, 2D, or 3D).

## Project overview

This project studies the structural dimensionality of organic–inorganic
antimony-halide compounds. We first identify relevant compounds in
chemistry papers and extract their compositions, organic cations,
structural information, and reported dimensionalities. These literature
data form the labeled dataset used throughout the project.

We train Random Forest (RF) and support vector machine (SVM) classifiers
to predict whether inorganic Sb-halide connectivity is zero-dimensional
or extended. We also evaluate GPT models on the same held-out task using
zero-shot, few-shot, and all-shot prompting.

To understand the predictions, we calculate permutation and SHAP
importance for the ML models and ask the GPT models to rank a set of
chemical and compositional features to understand feature importance for
each LLM. Finally, we apply the best-performing RF, SVM, and GPT
configurations to ten newly synthesized Sb-halide compounds.

## Repository layout

The directories follow the research workflow from literature extraction
to model evaluation and prediction:

-   `data_extraction/`: PDF-to-dataset pipeline with evidence-linked
    Excel and JSON export.
-   `data/`: prepared modeling datasets and the published train/test
    split used by the downstream workflows.
-   `ml_training/`: feature engineering, leakage-controlled splitting,
    Random Forest, and SVM training.
-   `feature_importance/`: permutation and SHAP importance for the best
    RF (main text) and best SVM (supplement).
-   `llm_prediction/`: held-out LLM evaluation and in-context prediction
    for new compounds.
-   `llm_feature_importance/`: paired feature-ranking experiments with
    and without labeled examples.
-   `model_prediction/`: feature generation and full-data RF and SVM
    prediction for new compounds.
-   `evaluation/`: ML/LLM comparisons and confusion-matrix plots.
-   `common/`: shared plotting utilities.
-   `results_ml/`: generated ML metrics, predictions, and fitted models.
-   `results_llm/`: LLM evaluation results behind Figure 3.
-   `figures/`: publication-figure scripts, generated publication
    figures, and diagnostic figures.

## Installation

Use Python 3.11 or 3.12:

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation pytorch-fast-transformers==0.4.0
```

The last command installs `pytorch-fast-transformers`, which SMI-TED
needs. It is installed separately because it is compiled against the
already-installed `torch`.

SMI-TED must be available when creating molecular embeddings. Supply its
`smi_ted_light` directory with `--smi-ted-dir`; the preparation script
can download the required files from Hugging Face if the argument is
omitted.

SMI-TED requires `transformers<5` (pinned in `requirements.txt`). From
`transformers` 5.0, `BertTokenizer` doesn't call SMI-TED's custom SMILES
tokenizer, so most SMILES tokens silently become `<pad>` and chemically
different cations receive identical embeddings.
`ml_training.prepare_data` stops with an error if this happens.

Set `OPENAI_API_KEY` before live data-extraction or LLM-prediction runs.

## Literature data extraction

The `data_extraction/` folder contains the code that builds the
literature-derived dataset used by the prediction workflows. It combines
deterministic PDF parsing, BM25 retrieval, validation, source citation
tracking, and final export with schema-constrained LLM stages for
scientific interpretation.

``` text
PDFs -> parse -> screen -> compound registry -> retrieve -> dossiers
     -> structured extraction -> verification -> final Excel + JSON datasets
```

Each extracted compound retains its supporting evidence text and
page-level source IDs. The pipeline produces one record per compound,
including reported formula and cation information, Sb-halide
connectivity, dimensionality and reasoning, synthesis evidence,
verification results, and automatic flags.

The output should be treated as raw, machine-extracted data rather than
a fully curated dataset. Users should review and correct the extracted
fields and supporting evidence against the original papers before using
the data for analysis.

The source articles are not included in this repository because they are
protected by copyright. The DOI of each compound's source paper is
listed in the `DOI` column of `data/merged_compound_cif_cation.xlsx`, so
you can download the PDFs yourself.

To run the extraction from the repository root:

1.  Enter the extraction directory and create the local environment
    file:

    ``` bash
    cd data_extraction
    cp .env.example .env
    ```

2.  Open `data_extraction/.env` and replace the placeholder with your
    OpenAI API key:

    ``` dotenv
    OPENAI_API_KEY="sk-..."
    ```

3.  Create the input directory and copy the article PDFs into it:

    ``` bash
    mkdir -p data/papers
    cp /path/to/your/pdfs/*.pdf data/papers/
    ```

    PDF filenames must contain their paper IDs, for example `P0001.pdf`.
    Nested directories under `data/papers/` are also supported.

4.  Run the complete pipeline for selected papers:

    ``` bash
    python -m src.run_pipeline all --paper-ids P0001,P0002
    ```

To process every PDF under `data/papers/`, run:

``` bash
python -m src.run_pipeline all
```

The final stage writes both formats directly:

``` text
data/output/final_dataset.xlsx
data/output/final_dataset.json
```

Intermediate JSONL artifacts and cached LLM responses are also created
under `data/`. Pipeline settings, model assignments, retrieval limits,
and input/output paths are defined in `config/pipeline.yaml`.

## ML workflow

The train/test split groups compounds by both source paper and organic
cation, so no paper and no cation structure appears in both sets. A few
chemical compositions do occur in both sets, but they correspond to
different organic cation structures, so they are distinct compounds
rather than duplicates.

The published split (321 training and 79 test compounds) is stored in
`data/train_test_split.csv`, one row per compound with its `compound_id`
and `split`. `ml_training.prepare_data` reuses this file by default, so
re-running it reproduces the committed `data/prepared_data.xlsx`
exactly. To draw a fresh grouped split instead, for example on a
different dataset, add `--new-split` (`--seed` sets its random seed).

``` bash
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

The commands above explain the best ML model overall (RF, inorganic
features plus SMI-TED embeddings with PCA-80; main text). For the best
SVM (supplement), add `--model svm` and write to a separate folder:

``` bash
python -m feature_importance.permutation_importance \
  --data data/prepared_data.xlsx \
  --results results_ml/ml_results.csv \
  --model svm --out results_ml/svm --figures figures/svm

python -m feature_importance.shap_importance \
  --data data/prepared_data.xlsx \
  --results results_ml/ml_results.csv \
  --model svm --out-csv results_ml/svm/shap_importance.csv --figures figures/svm
```

To predict the dimensionality of the ten newly synthesized mixed-halide
Sb compounds (`data/new_mixed_halide_sb_compounds.xlsx`):

``` bash
python -m model_prediction.featurize_new_compounds \
  --input data/new_mixed_halide_sb_compounds.xlsx \
  --smi-ted-dir /path/to/smi_ted_light \
  --output data/new_compound_features.csv

python -m model_prediction.retrain_and_predict \
  --data data/prepared_data.xlsx \
  --new-features data/new_compound_features.csv \
  --out results_ml/new_compounds
```

`retrain_and_predict` refits the best configuration of each model family
(RF and SVM, by test macro-F1 in `results_ml/ml_results.csv`) on all 400
labeled compounds and writes both models' predictions to
`results_ml/new_compounds/predictions.csv`, for the ten compounds
(`NEW_01`–`NEW_10`). The training data and the new compounds must be
featurized with the same SMI-TED setup (`transformers<5`).

## LLM prediction

`evaluate_llms` tests five GPT models on the held-out test set (the same
79 compounds as the ML models) under zero-shot, 12-shot, and all-shot
prompting (the 321 training compounds as examples), with the cation
described either by its IUPAC name or by its SMILES. Its results are
behind [Figure
3](#figure-3-llm-performance-across-prompting-conditions-and-cation-representations).
`predict_new_compounds` then applies the best configuration (GPT-5.1,
all-shot, IUPAC names) to the ten new compounds, using all 400 labeled
compounds as examples; run the new-compound step of the [ML
workflow](#ml-workflow) first, since it reads
`data/new_compound_features.csv`. Both need `OPENAI_API_KEY` (see
[Installation](#installation)).

``` bash
python -m llm_prediction.evaluate_llms \
  --data data/prepared_data.xlsx \
  --out results_llm

python -m llm_prediction.predict_new_compounds \
  --data data/prepared_data.xlsx \
  --new-features data/new_compound_features.csv \
  --output results_llm/new_compound_predictions.csv
```

## LLM feature importance

This experiment asks each of four LLMs (GPT-5.1, GPT-5.5, GPT-5.6 Sol,
and GPT-6 Astra) to rank seven feature categories, such as the halide:Sb
ratio and the organic-cation structure, from most to least important for
predicting whether a compound is 0D or non-0D. Each model ranks them 30
times, with the categories listed in a different order each time so that
the order they appear in does not bias the ranking. Each ranking is
converted to Borda points (7 − rank: 6 for the first-ranked category, 0
for the last), divided by 6 to give a normalized Borda score between 0
and 1, and averaged over the 30 trials.

The experiment is run twice, using the same 30 orders:

-   **without examples**: the models see only the feature descriptions
    (results in `llm_feature_importance/results/`); and
-   **with examples**: the models also see the 321 training compounds
    with their known dimensionality (results in
    `llm_feature_importance/results_with_data/`).

Every individual ranking (model, trial, feature, rank, and normalized
Borda score) is in `rankings_long.csv` in each of these two folders.

The "with examples" results are panel (b) of [Figure
4](#figure-4-feature-importance-for-the-best-performing-ml-and-llm-configurations).
The "without examples" results, and how the rankings change when
examples are added (the `compare` step), make up [Figure
5](#figure-5-llm-feature-importance-without-training-examples-and-the-withwithout-data-difference).

``` bash
python -m unittest -v llm_feature_importance.tests

python -m llm_feature_importance.collect_paired \
  --data data/prepared_data.xlsx

python -m llm_feature_importance.analyze --out llm_feature_importance/results
python -m llm_feature_importance.analyze --out llm_feature_importance/results_with_data
python -m llm_feature_importance.plot --out llm_feature_importance/results
python -m llm_feature_importance.plot --out llm_feature_importance/results_with_data
python -m llm_feature_importance.compare \
  --no-data llm_feature_importance/results \
  --with-data llm_feature_importance/results_with_data \
  --out llm_feature_importance/comparison
```

## Reproducing manuscript figures

### Figure 2: dataset composition and organic-cation chemical space

Figure 2 contains:

-   **(a)** the 0D versus non-0D dimensionality distribution;
-   **(b)** the dimensionality distribution within the F, Cl, Br, and I
    families; and
-   **(c)** the RDKit-descriptor UMAP of the original and newly proposed
    organic cations.

Run from the repository root, after completing
[Installation](#installation). `figures/figrue_2/figure_2_reference.png`
shows the expected result, so you can compare it against your own
output.

#### 1. Regenerate the UMAP coordinates for panel (c) (optional)

Panel (c)'s input coordinates are already in
`figures/figrue_2/reference_coordinates/`, so you can skip this step and
go straight to step 2. Run it if you want to regenerate those
coordinates yourself — to reproduce them from scratch, or to add new
proposed cations. `requirements.txt` pins the UMAP-relevant package
versions (`numpy`, `scipy`, `umap-learn`, `matplotlib`), so installing
it as specified reproduces `reference_coordinates/` exactly.

``` bash
python figures/figrue_2/generate_umap_coordinates.py
```

This reads `data/merged_compound_cif_cation.xlsx` and
`figures/figrue_2/new_cations.csv`, then writes
`umap_original_cations.csv` and `umap_new_cations.csv` to
`figures/figrue_2/generated_coordinates/`.

#### 2. Generate Figure 2

``` bash
python figures/figrue_2/make_figure_2.py
```

This plots from `figures/figrue_2/reference_coordinates/` by default. If
you regenerated coordinates in step 1 and want to plot from those
instead, pass them explicitly:

``` bash
python figures/figrue_2/make_figure_2.py \
  --coordinates-dir figures/figrue_2/generated_coordinates
```

Both the combined figure (`figure_2.png`/`.pdf`) and the individual
panels are written to `figures/figrue_2/output/`.

### Figure 3: LLM performance across prompting conditions and cation representations {#figure-3-llm-performance-across-prompting-conditions-and-cation-representations}

Figure 3 shows test macro-F1 for five LLMs under zero-shot, 12-shot, and
all-shot prompting, using either IUPAC cation names or SMILES as input,
against a dashed best-ML reference line.
`figures/figure_3/figure_3_reference.png` shows the expected result, so
you can compare it against your own output.

Figure 3 is built from LLM evaluation results, which are expensive to
regenerate (live API calls, billed per token) and not guaranteed to
reproduce exactly on a rerun (see caveats below), so
`results_llm/llm_results.csv` is committed as the reference data behind
the published figure.

To generate Figure 3 from that reference data:

``` bash
python figures/figure_3/make_figure_3.py
```

Figure 3 (`figure_3.png`/`.pdf`) is written to
`figures/figure_3/output/`. The dashed reference line is the best ML
test macro-F1 (0.714, RF); pass `--ml-baseline` to change it.

`python -m evaluation.plot_llm_results` draws the same figure
(`figures/llm_results_by_representation_final.png`) together with
several alternative/diagnostic views of the same results, not used in
the manuscript.

#### Regenerating the LLM evaluation results (optional)

Only do this if you want to re-run the LLM evaluation yourself rather
than use the committed reference results. It requires an OpenAI API key
with access to the five evaluated models:

``` bash
export OPENAI_API_KEY="sk-..."
python -m llm_prediction.evaluate_llms \
  --data data/prepared_data.xlsx \
  --out results_llm
```

Then point `make_figure_3.py` at the new output:

``` bash
python figures/figure_3/make_figure_3.py --llm-results results_llm/llm_results.csv
```

A rerun is not guaranteed to reproduce the published numbers exactly,
for reasons outside this code's control:

-   Three of the five models are pinned to a dated snapshot
    (`gpt-4.1-2025-04-14`, `gpt-5.1-2025-11-13`, `gpt-5.5-2026-04-23`);
    the other two (`gpt-5.6-sol`, `gpt-6-astra`) are undated aliases, so
    OpenAI may change which underlying model they resolve to over time.
-   Except for `gpt-4.1`, these are reasoning models, whose outputs are
    not guaranteed to be bit-for-bit reproducible across calls even on
    an unchanged snapshot with identical prompts.

### Figure 4: feature importance for the best-performing ML and LLM configurations

Figure 4 contains:

-   **(a)** permutation importance for the best ML model (Random Forest
    with inorganic features plus SMI-TED embeddings with PCA-80); and
-   **(b)** feature importance for each LLM's best setting, summarized
    by Borda score, from the "with examples" condition of the [LLM
    feature importance](#llm-feature-importance) experiment above.

`figures/figure_4/figure_4_reference.png` shows the expected result.

Panel (a) is produced by the `feature_importance.permutation_importance`
step in [ML workflow](#ml-workflow) above — run that first if you
haven't already. It's fully deterministic (fixed random seeds, pinned
`scikit-learn`), so no reference data is needed; just rerun it.

Panel (b) needs
`llm_feature_importance.plot --out llm_feature_importance/results_with_data`,
from [LLM feature importance](#llm-feature-importance) above. That
command's input (`model_feature_importance.csv`,
`model_stability_kendall.csv`, `intermodel_kendall.csv`) is committed as
reference data in `llm_feature_importance/results_with_data/`, since it
comes from live, costly LLM calls (the same reproducibility caveats as
Figure 3 apply if you regenerate it yourself via `collect_paired` and
`analyze`). `rankings_long.csv` is also committed there — the raw
per-trial LLM output (one row per model/trial/feature, with its rank and
Borda score) that the other three files are aggregated from.

Once both panels exist at their default paths
(`figures/ml_permutation_importance_no_title.png` and
`llm_feature_importance/results_with_data/figures/llm_feature_importance_heatmap.png`),
compose them into Figure 4:

``` bash
python figures/figure_4/make_figure_4.py
```

The combined figure (`figure_4.png`/`.pdf`) is written to
`figures/figure_4/output/`.

The same command also writes the supplementary figure
(`figure_4_supplement_svm.png`/`.pdf`): permutation importance for the
best SVM (inorganic features plus SMI-TED embeddings with PCA-80), in
the same style as panel (a). It is drawn from
`results_ml/svm/permutation_importance.csv`; if that file does not
exist, the script first runs
`feature_importance.permutation_importance --model svm` (which needs
`results_ml/ml_results.csv` and the trained models from the [ML
workflow](#ml-workflow)).
`figures/figure_4/figure_4_supplement_svm_reference.png` shows the
expected result.

### Figure 5: LLM feature importance without training examples, and the with/without-data difference

Figure 5 contains:

-   **(a)** LLM-assessed feature importance without training examples,
    from the "no examples" condition of the [LLM feature
    importance](#llm-feature-importance) experiment above; and
-   **(b)** the change in feature importance between the with-data and
    without-data settings, from that experiment's `compare` step.

`figures/figure_5/figure_5_reference.png` shows the expected result.

Panel (a) needs
`llm_feature_importance.plot --out llm_feature_importance/results`, and
panel (b) needs `llm_feature_importance.compare` (both from [LLM feature
importance](#llm-feature-importance) above). Both read from CSVs —
`model_feature_importance.csv`, `model_stability_kendall.csv`,
`intermodel_kendall.csv`, `rankings_long.csv`, and `run_manifest.json` —
committed as reference data in `llm_feature_importance/results/` (this
figure's no-data condition) and
`llm_feature_importance/results_with_data/` (already used by Figure 4),
for the same reason as Figure 3 and 4's reference data: they come from
live, costly LLM calls. Unlike collection, `compare` itself is a
deterministic computation over those committed CSVs — no API calls — so
it always reproduces exactly from them.

Once both panels exist at their default paths
(`llm_feature_importance/results/figures/llm_feature_importance_heatmap.png`
and `llm_feature_importance/comparison/comparison_delta_heatmap.png`),
compose them into Figure 5:

``` bash
python figures/figure_5/make_figure_5.py
```

The combined figure (`figure_5.png`/`.pdf`) is written to
`figures/figure_5/output/`.
