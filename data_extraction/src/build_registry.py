"""Build the compound registry and resolve aliases within each paper.

Reads eligible candidates from 03_eligibility.jsonl plus the paper's
Sb-relevant chunks, and asks the LLM to group labels/names/formulas/aliases
into distinct compounds. `compound_id` is assigned deterministically in code
(first-mention order by source order_index), not by the model, so IDs are
stable across reruns. Writes data/registry/04_compound_registry.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import CompoundRegistryEntry
from src.llm_client import LLMClient
from src.screen_compounds import select_sb_chunks
from src.utils import get_logger, load_config, load_prompt, read_jsonl, write_jsonl

logger = get_logger("build_registry")
PROMPT_VERSION = "v1"

REGISTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "compounds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "labels_in_paper": {"type": "array", "items": {"type": "string"}},
                    "compound_name_reported": {"type": ["string", "null"]},
                    "compound_formula_reported": {"type": ["string", "null"]},
                    "supporting_source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "labels_in_paper",
                    "compound_name_reported",
                    "compound_formula_reported",
                    "supporting_source_ids",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["compounds"],
    "additionalProperties": False,
}


def _stub(payload: dict) -> dict:
    return {"compounds": []}


def build_paper_registry(
    client: LLMClient, model: str, paper_id: str, units: list[dict], eligible_labels: list[str]
) -> list[CompoundRegistryEntry]:
    sb_units = select_sb_chunks(units)
    order_by_source = {u["source_id"]: u["order_index"] for u in units}

    system_prompt = load_prompt("02_compound_registry.md")
    user_payload = {
        "task": "compound_registry",
        "paper_id": paper_id,
        "eligible_candidate_labels": eligible_labels,
        "instructions": "Resolve aliases and assemble one registry entry per distinct eligible compound.",
        "evidence_units": [
            {"source_id": u["source_id"], "section": u.get("section"), "text": u["text"]} for u in sb_units
        ],
        "output_schema": REGISTRY_SCHEMA,
    }
    result = client.call_structured(
        task="compound_registry",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=REGISTRY_SCHEMA,
        schema_name="compound_registry",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_stub,
    )

    raw_entries = result.data.get("compounds", [])
    def first_seen(entry: dict) -> int:
        idxs = [order_by_source[sid] for sid in entry.get("supporting_source_ids", []) if sid in order_by_source]
        return min(idxs) if idxs else 10**9

    raw_entries.sort(key=first_seen)

    entries = []
    for i, e in enumerate(raw_entries, start=1):
        entries.append(
            CompoundRegistryEntry(
                paper_id=paper_id,
                compound_id=f"{paper_id}_C{i}",
                labels_in_paper=e["labels_in_paper"],
                compound_name_reported=e["compound_name_reported"],
                compound_formula_reported=e["compound_formula_reported"],
                supporting_source_ids=e["supporting_source_ids"],
                model=result.model,
                prompt_version=result.prompt_version,
                created_at=result.created_at,
            )
        )
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the per-paper Sb-compound registry.")
    ap.add_argument("--paper-ids", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    model = cfg["llm"]["registry_model"]

    units_by_paper: dict[str, list[dict]] = defaultdict(list)
    for u in read_jsonl(data_dir / "chunks" / "02_chunks.jsonl"):
        units_by_paper[u["paper_id"]].append(u)

    eligible_by_paper: dict[str, list[str]] = defaultdict(list)
    for d in read_jsonl(data_dir / "registry" / "03_eligibility.jsonl"):
        if d["is_eligible"]:
            eligible_by_paper[d["paper_id"]].append(d["candidate_label"])

    paper_ids = sorted(eligible_by_paper.keys())
    if args.paper_ids:
        wanted = {p.strip().upper() for p in args.paper_ids.split(",")}
        paper_ids = [p for p in paper_ids if p in wanted]

    client = LLMClient(provider=cfg["llm"]["provider"], dry_run=args.dry_run)
    all_entries: list[CompoundRegistryEntry] = []
    for paper_id in paper_ids:
        if not eligible_by_paper[paper_id]:
            logger.info("skip %s: no eligible candidates", paper_id)
            continue
        logger.info("building registry for %s (%d eligible labels)", paper_id, len(eligible_by_paper[paper_id]))
        entries = build_paper_registry(client, model, paper_id, units_by_paper[paper_id], eligible_by_paper[paper_id])
        all_entries.extend(entries)
        logger.info("  -> %d compounds", len(entries))

    write_jsonl(data_dir / "registry" / "04_compound_registry.jsonl", (e.model_dump() for e in all_entries))
    logger.info("wrote %d compound registry entries", len(all_entries))


if __name__ == "__main__":
    main()
