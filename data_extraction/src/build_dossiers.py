"""Build source-tracked evidence dossiers for retrieved passages.

Consolidates the relevant retrieval hits for each (paper, compound, category)
into one source-tracked evidence text. Writes data/dossiers/06_evidence_dossiers.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import EvidenceDossier
from src.llm_client import LLMClient
from src.utils import get_logger, load_config, load_prompt, read_jsonl, write_jsonl

logger = get_logger("build_dossiers")
PROMPT_VERSION = "v1"
CATEGORY_MODEL_KEY = {"cation": "dossier_model", "connectivity": "dossier_model", "synthesis": "dossier_model"}

DOSSIER_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_text": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "not_reported": {"type": "boolean"},
    },
    "required": ["evidence_text", "source_ids", "not_reported"],
    "additionalProperties": False,
}


def _stub(payload: dict) -> dict:
    return {"evidence_text": "", "source_ids": [], "not_reported": True}


def build_dossier(
    client: LLMClient,
    model: str,
    paper_id: str,
    compound_id: str,
    category: str,
    relevant_units: list[dict],
    aliases: list[str] | None = None,
) -> EvidenceDossier:
    aliases = aliases or []
    system_prompt = load_prompt("04_build_dossier.md")
    user_payload = {
        "task": "build_dossier",
        "paper_id": paper_id,
        "compound_id": compound_id,
        "compound_aliases": aliases,
        "target_category": category,
        "instructions": (
            f"Construct the {category} evidence text for the compound called {aliases!r} in this "
            f"paper (internal id {compound_id}) using only these units. Some supplied units may "
            "actually describe a different, similarly-named compound in the same paper (this paper "
            "reports several related compounds) — exclude any sentence that is about a different "
            "numbered compound, halide, or formula than the one named in compound_aliases."
        ),
        "evidence_units": [{"source_id": u["source_id"], "text": u["text"]} for u in relevant_units],
        "output_schema": DOSSIER_SCHEMA,
    }
    result = client.call_structured(
        task="build_dossier",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=DOSSIER_SCHEMA,
        schema_name="evidence_dossier",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_stub,
    )
    d = result.data
    return EvidenceDossier(
        paper_id=paper_id,
        compound_id=compound_id,
        target_category=category,
        evidence_text=d["evidence_text"],
        source_ids=d["source_ids"],
        not_reported=d["not_reported"],
        model=result.model,
        prompt_version=result.prompt_version,
        created_at=result.created_at,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build compound-specific evidence dossiers.")
    ap.add_argument("--paper-ids", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    model = cfg["llm"]["dossier_model"]
    rerank_top_k = cfg["retrieval"]["rerank_top_k"]

    units_by_paper: dict[str, dict[str, dict]] = defaultdict(dict)
    for u in read_jsonl(data_dir / "chunks" / "02_chunks.jsonl"):
        units_by_paper[u["paper_id"]][u["source_id"]] = u

    aliases_by_compound: dict[tuple[str, str], list[str]] = {}
    for c in read_jsonl(data_dir / "registry" / "04_compound_registry.jsonl"):
        a = [c.get("compound_formula_reported"), c.get("compound_name_reported")] + c.get("labels_in_paper", [])
        aliases_by_compound[(c["paper_id"], c["compound_id"])] = [x for x in a if x]

    hits_by_group: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for h in read_jsonl(data_dir / "retrieval" / "05_retrieval_hits.jsonl"):
        if h.get("is_relevant"):
            key = (h["paper_id"], h["compound_id"], h["target_category"])
            hits_by_group[key].append(h)

    paper_ids = sorted({k[0] for k in hits_by_group})
    if args.paper_ids:
        wanted = {p.strip().upper() for p in args.paper_ids.split(",")}
        paper_ids = [p for p in paper_ids if p in wanted]

    client = LLMClient(provider=cfg["llm"]["provider"], dry_run=args.dry_run)
    dossiers: list[EvidenceDossier] = []
    for (paper_id, compound_id, category), hits in sorted(hits_by_group.items()):
        if paper_id not in paper_ids:
            continue
        hits_sorted = sorted(hits, key=lambda h: -h["score"])[:rerank_top_k]
        relevant_units = [units_by_paper[paper_id][h["source_id"]] for h in hits_sorted if h["source_id"] in units_by_paper[paper_id]]
        logger.info("dossier %s / %s / %s (%d units)", paper_id, compound_id, category, len(relevant_units))
        aliases = aliases_by_compound.get((paper_id, compound_id), [])
        dossiers.append(build_dossier(client, model, paper_id, compound_id, category, relevant_units, aliases))

    write_jsonl(data_dir / "dossiers" / "06_evidence_dossiers.jsonl", (d.model_dump() for d in dossiers))
    logger.info("wrote %d dossiers", len(dossiers))


if __name__ == "__main__":
    main()
