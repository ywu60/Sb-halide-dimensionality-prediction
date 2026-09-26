"""CLI orchestrator — runs pipeline stages in order (plan §6.1, §12).

Each stage is deterministic Python that reads a versioned JSONL artifact and
writes the next one; this script just calls them in sequence with a shared
`--paper-ids` / `--dry-run` selection so a pilot run touches exactly the
papers you name. It does not itself call any LLM or parsing logic — that
would duplicate the per-stage scripts, which remain independently runnable.

Usage:
    python -m src.run_pipeline all --paper-ids P0001 --dry-run
    python -m src.run_pipeline all --paper-ids P0001,P0002
    python -m src.run_pipeline parse --paper-ids P0001
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGES = [
    # (name, module, needs_llm_flags, accepts_paper_ids)
    ("parse", "src.parse_documents", False, True),
    ("screen", "src.screen_compounds", True, True),
    ("registry", "src.build_registry", True, True),
    ("retrieve", "src.retrieve_evidence", True, True),
    ("dossier", "src.build_dossiers", True, True),
    ("extract", "src.extract_records", True, True),
    ("verify", "src.verify_records", True, True),
    ("export", "src.export_final_dataset", False, False),
]


def run_stage(module: str, paper_ids: str | None, dry_run: bool, needs_llm_flags: bool, accepts_paper_ids: bool) -> None:
    cmd = [sys.executable, "-m", module]
    if paper_ids and accepts_paper_ids:
        cmd += ["--paper-ids", paper_ids]
    if dry_run and needs_llm_flags:
        cmd += ["--dry-run"]
    print(f"\n=== {module} ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Sb-halide extraction pipeline.")
    ap.add_argument(
        "stage",
        choices=["all"] + [s[0] for s in STAGES],
        help="'all' runs parse through direct final-dataset export in sequence",
    )
    ap.add_argument("--paper-ids", type=str, default=None, help="Comma-separated paper IDs, e.g. P0001,P0002")
    ap.add_argument("--dry-run", action="store_true", help="Use stubbed LLM calls, no API key / cost")
    args = ap.parse_args()

    if args.stage == "all":
        for name, module, needs_llm_flags, accepts_paper_ids in STAGES:
            run_stage(module, args.paper_ids, args.dry_run, needs_llm_flags, accepts_paper_ids)
        print(
            "\nDone. Final datasets: data/output/final_dataset.xlsx "
            "and data/output/final_dataset.json"
        )
        return

    name, module, needs_llm_flags, accepts_paper_ids = next(s for s in STAGES if s[0] == args.stage)
    run_stage(module, args.paper_ids, args.dry_run, needs_llm_flags, accepts_paper_ids)


if __name__ == "__main__":
    main()
