# Sb-Halide Literature Data Extraction Framework

This project converts a collection of antimony-halide research papers into a
structured, evidence-linked dataset in Excel and JSON. Its unit of output is one compound,
not one paper: if a paper reports several eligible compounds, the final dataset
contains one row for each compound.

The framework combines deterministic document processing with schema-constrained
LLM tasks. Python handles PDF parsing, lexical retrieval, validation, provenance,
and export. LLMs are used only where scientific interpretation is needed, such
as identifying reported compounds, consolidating evidence, extracting cation
descriptions, and reasoning about Sb-halide dimensionality. Every extracted
claim retains the source IDs of the page-level passages that support it.

The final spreadsheet includes:

- reported compound names, labels, and formulas;
- organic-cation names, abbreviations, formulas, and evidence;
- Sb oxidation state, halides, Sb-halide units, and connectivity evidence;
- Sb-halide dimensionality (`0D`, `1D`, `2D`, `3D`, or `Unknown`) with reasoning;
- synthesis evidence;
- verification results, automatic quality flags, and source citations.

The pipeline exports verified results directly to
`data/output/final_dataset.xlsx` and `data/output/final_dataset.json`. Human
review is optional and happens by opening the workbook and inspecting or editing cells. There is no required
review-status field or separate apply-review step.

## Framework workflow

```text
PDF articles
  -> layout-aware parsing and table extraction
  -> paper eligibility screening
  -> per-paper compound registry
  -> compound-conditioned evidence retrieval
  -> evidence dossier construction
  -> structured cation and dimensionality extraction
  -> evidence-based verification and deterministic checks
  -> final Excel and JSON datasets
```

Each stage reads a versioned JSONL artifact and writes the next artifact. This
makes runs restartable and allows intermediate decisions to be inspected without
rerunning the entire pipeline. LLM responses are cached by model, prompt version,
and input hash to avoid paying twice for unchanged work.

## Repository layout

```text
config/       Pipeline settings, retrieval queries, and dimensionality rules
prompts/      Versioned instructions for each LLM task
schemas/      Pydantic models for documents, compounds, evidence, and records
src/          Independently runnable pipeline stages and shared utilities
tests/        Schema, provenance, validation, and export tests
figures/      Extraction-workflow figure and its generator
data/         Generated artifacts and final output (ignored by Git)
```

The stage orchestrator is `src/run_pipeline.py`. LLM access is isolated in
`src/llm_client.py`; parsing, indexing, validation, and final export do not call
an LLM.

## Current scope and limitations

- PDF parsing uses PyMuPDF text blocks and native table detection. It does not
  OCR image-only scans. Docling or an OCR stage would improve difficult layouts.
- Retrieval uses BM25, exact compound aliases, and neighboring passages. A dense
  retrieval interface exists but is not connected to an embedding model.
- The current parser processes main articles only. Supporting-information files
  are not yet incorporated.
- The configured LLM provider is OpenAI and structured outputs use strict JSON
  schemas.
- Verification can perform bounded, targeted re-retrieval for failed cation,
  connectivity, or synthesis checks.
- A dimensionality other than `Unknown` requires sufficient evidence under the
  configured dimensionality rules.

## Setup

Create an environment and install the dependencies:

```bash
cd data_extraction
python -m pip install -r requirements.txt
```

Copy the environment template and add an API key for live LLM runs:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY="sk-..."
```

The included configuration expects the local corpus at
`../../analysis 3/Antimony papers`, relative to this project. Change `paths.papers_dir` in
`config/pipeline.yaml` when using a different corpus location. PDFs are read in
place and are never copied into the project. Generated intermediates and Excel
outputs are written under `data/`.

## Running the pipeline

Start with a dry run to validate the complete workflow without an API key or
API cost:

```bash
python -m src.run_pipeline all --paper-ids P0001 --dry-run
```

Run a live pilot on a small set of papers before scaling up:

```bash
python -m src.run_pipeline all --paper-ids P0001,P0002
```

Omitting `--paper-ids` processes every paper discovered under `papers_dir`.
For a large corpus, make this choice deliberately.

The completed run writes:

```text
data/output/final_dataset.xlsx
data/output/final_dataset.json
```

The workbook contains a filterable `Final Dataset` sheet with frozen headers,
one compound per row, flattened list fields, evidence text, and source IDs. The
JSON file contains the same final fields while preserving list values as JSON
arrays. Rerunning the export regenerates both files from verified JSONL records,
so keep a separate copy if you manually edit the workbook.

## Running individual stages

Stages can be restarted independently:

```bash
python -m src.run_pipeline parse --paper-ids P0001
python -m src.run_pipeline verify --paper-ids P0001
python -m src.run_pipeline export
```

Available stages are `parse`, `screen`, `registry`, `retrieve`, `dossier`,
`extract`, `verify`, and `export`.

## Cost control

- Use `--dry-run` to test orchestration and schemas without LLM calls.
- Always use `--paper-ids` for pilot runs.
- Cached responses under `data/.cache/` are reused when the model, prompt, and
  input have not changed.
- Model assignments and retrieval limits are configured in
  `config/pipeline.yaml`.

## Tests

```bash
pytest tests/ -v
```

The tests cover schema constraints, dimensionality rules, source-ID integrity,
text-quality flags, chemistry consistency checks, and final-dataset export.

## Workflow figure

Regenerate the vector and raster workflow figures with:

```bash
python figures/make_pipeline_figure.py
```
