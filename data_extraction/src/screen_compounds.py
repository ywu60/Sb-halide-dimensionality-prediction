"""Screen papers for eligible Sb-halide compounds.

Reads data/chunks/02_chunks.jsonl, filters to Sb-relevant evidence units per
paper, and asks the LLM to enumerate every candidate material with an
eligibility decision. Writes data/registry/03_eligibility.jsonl.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import EligibilityDecision
from src.llm_client import LLMClient
from src.utils import get_logger, load_config, load_prompt, read_jsonl, write_jsonl

logger = get_logger("screen_compounds")
PROMPT_VERSION = "v1"
SB_PATTERN = re.compile(r"\bSb\b|antimon", re.IGNORECASE)

DECISION_CATEGORIES = [
    "stoichiometric_sb_compound",
    "doped_or_trace_sb",
    "device_or_composite",
    "non_sb_precursor",
    "comparison_analogue",
    "background_only",
    "other_excluded",
]

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_label": {"type": "string"},
                    "is_eligible": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "decision_category": {"type": "string", "enum": DECISION_CATEGORIES},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["candidate_label", "is_eligible", "reason", "decision_category", "source_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def select_sb_chunks(units: list[dict]) -> list[dict]:
    return [u for u in units if SB_PATTERN.search(u["text"])]


def _stub(payload: dict) -> dict:
    return {"candidates": []}


def screen_paper(client: LLMClient, model: str, paper_id: str, units: list[dict]) -> list[EligibilityDecision]:
    sb_units = select_sb_chunks(units)
    if not sb_units:
        logger.warning("no Sb-related evidence units found for %s", paper_id)
        return []

    system_prompt = load_prompt("01_eligibility.md")
    user_payload = {
        "task": "eligibility_screening",
        "paper_id": paper_id,
        "instructions": "Identify every candidate material and decide eligibility per the rules above.",
        "evidence_units": [
            {"source_id": u["source_id"], "section": u.get("section"), "text": u["text"]} for u in sb_units
        ],
        "output_schema": CANDIDATES_SCHEMA,
    }
    result = client.call_structured(
        task="eligibility_screening",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=CANDIDATES_SCHEMA,
        schema_name="eligibility_candidates",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_stub,
    )

    decisions = []
    for c in result.data.get("candidates", []):
        decisions.append(
            EligibilityDecision(
                paper_id=paper_id,
                candidate_label=c["candidate_label"],
                is_eligible=c["is_eligible"],
                reason=c["reason"],
                decision_category=c["decision_category"],
                source_ids=c["source_ids"],
                model=result.model,
                prompt_version=result.prompt_version,
                created_at=result.created_at,
            )
        )
    return decisions


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen candidate materials for Sb-halide eligibility.")
    ap.add_argument("--paper-ids", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    model = cfg["llm"]["eligibility_model"]

    units_by_paper: dict[str, list[dict]] = defaultdict(list)
    for u in read_jsonl(data_dir / "chunks" / "02_chunks.jsonl"):
        units_by_paper[u["paper_id"]].append(u)

    paper_ids = sorted(units_by_paper.keys())
    if args.paper_ids:
        wanted = {p.strip().upper() for p in args.paper_ids.split(",")}
        paper_ids = [p for p in paper_ids if p in wanted]

    client = LLMClient(provider=cfg["llm"]["provider"], dry_run=args.dry_run)
    all_decisions: list[EligibilityDecision] = []
    for paper_id in paper_ids:
        logger.info("screening %s (%d Sb-related units)", paper_id, len(select_sb_chunks(units_by_paper[paper_id])))
        decisions = screen_paper(client, model, paper_id, units_by_paper[paper_id])
        all_decisions.extend(decisions)
        logger.info("  -> %d candidates, %d eligible", len(decisions), sum(d.is_eligible for d in decisions))

    write_jsonl(data_dir / "registry" / "03_eligibility.jsonl", (d.model_dump() for d in all_decisions))
    logger.info("wrote %d eligibility decisions", len(all_decisions))


if __name__ == "__main__":
    main()
