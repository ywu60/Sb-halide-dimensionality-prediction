"""Retrieve compound-specific evidence from each paper.

For every (paper, compound, category) this stage:
  1. builds a query bundle from the fixed vocabulary in retrieval_queries.yaml
     plus the compound's aliases;
  2. runs BM25 retrieval restricted to that paper, unions with exact
     alias-substring matches, and adds neighboring evidence units;
  3. sends the candidate pool to an LLM relevance classifier (prompts/03);
  4. writes every candidate (relevant or not) as a RetrievalHit, so the
     dossier stage and any human auditor can see what was rejected and why.

Writes data/retrieval/05_retrieval_hits.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import RetrievalHit
from src.build_index import BM25Index
from src.llm_client import LLMClient
from src.utils import get_logger, load_config, load_prompt, load_yaml, read_jsonl, write_jsonl

logger = get_logger("retrieve_evidence")
PROMPT_VERSION = "v1"
ALL_CATEGORIES = ["cation", "connectivity", "synthesis"]

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "is_relevant": {"type": "boolean"},
                    "relevance_type": {"type": ["string", "null"]},
                },
                "required": ["source_id", "is_relevant", "relevance_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relevance"],
    "additionalProperties": False,
}


def _stub(payload: dict) -> dict:
    return {
        "relevance": [
            {"source_id": u["source_id"], "is_relevant": True, "relevance_type": "dry_run_stub"}
            for u in payload["evidence_units"]
        ]
    }


def gather_candidates(
    index: BM25Index,
    units_by_id: dict[str, dict],
    query_terms: list[str],
    aliases: list[str],
    bm25_top_k: int,
    neighbor_window: int,
) -> dict[str, str]:
    scored: dict[str, float] = {}
    method: dict[str, str] = {}

    for q in query_terms:
        for unit, score in index.query(q, bm25_top_k):
            sid = unit["source_id"]
            if sid not in scored or score > scored[sid]:
                scored[sid] = score
                method[sid] = "lexical_bm25"

    alias_terms = [a.lower() for a in aliases if a and len(a) >= 2]
    if alias_terms:
        for u in index.units:
            low = u["text"].lower()
            if any(a in low for a in alias_terms):
                sid = u["source_id"]
                if sid not in scored:
                    scored[sid] = max(scored.values(), default=1.0) + 1.0
                    method[sid] = "alias_match"

    for sid in list(scored.keys()):
        u = units_by_id.get(sid)
        if not u:
            continue
        for _ in range(neighbor_window):
            for nb_sid in (u.get("prev_source_id"), u.get("next_source_id")):
                if nb_sid and nb_sid not in scored:
                    scored[nb_sid] = 0.0
                    method[nb_sid] = "neighbor"

    return scored, method


def classify_relevance(
    client: LLMClient,
    model: str,
    paper_id: str,
    compound_id: str,
    aliases: list[str],
    category: str,
    candidate_units: list[dict],
) -> dict[str, tuple[bool, str | None]]:
    if not candidate_units:
        return {}
    system_prompt = load_prompt("03_evidence_relevance.md")
    user_payload = {
        "task": "evidence_relevance",
        "paper_id": paper_id,
        "compound_id": compound_id,
        "compound_aliases": aliases,
        "target_category": category,
        "instructions": (
            f"Classify relevance to the '{category}' category for the compound called "
            f"{aliases!r} in this paper (internal id {compound_id}). This paper reports "
            "multiple similar compounds — reject any passage that concerns a different "
            "compound (a different numbered label, a different halide, a different formula) "
            "even if it is structurally very similar."
        ),
        "evidence_units": [{"source_id": u["source_id"], "text": u["text"]} for u in candidate_units],
        "output_schema": RELEVANCE_SCHEMA,
    }
    result = client.call_structured(
        task="evidence_relevance",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=RELEVANCE_SCHEMA,
        schema_name="evidence_relevance",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_stub,
    )
    return {r["source_id"]: (r["is_relevant"], r["relevance_type"]) for r in result.data.get("relevance", [])}


def retrieve_for_compound(
    client: LLMClient,
    model: str,
    cfg: dict,
    query_concepts: dict,
    paper_id: str,
    compound: dict,
    units: list[dict],
    categories: list[str] = ALL_CATEGORIES,
) -> list[RetrievalHit]:
    units_by_id = {u["source_id"]: u for u in units}
    index = BM25Index(units)
    aliases = [compound.get("compound_formula_reported"), compound.get("compound_name_reported")]
    aliases += compound.get("labels_in_paper", [])
    aliases = [a for a in aliases if a]

    bm25_top_k = cfg["retrieval"]["bm25_top_k"]
    neighbor_window = cfg["retrieval"]["neighbor_window"]
    pool_cap = cfg["retrieval"]["classification_pool_cap"]

    hits: list[RetrievalHit] = []
    for category in categories:
        query_terms = list(query_concepts[category]) + aliases
        scored, method = gather_candidates(index, units_by_id, query_terms, aliases, bm25_top_k, neighbor_window)
        ranked_sids = sorted(scored.keys(), key=lambda s: -scored[s])[:pool_cap]
        candidate_units = [units_by_id[s] for s in ranked_sids]

        relevance = classify_relevance(client, model, paper_id, compound["compound_id"], aliases, category, candidate_units)

        for sid in ranked_sids:
            is_rel, rel_type = relevance.get(sid, (None, None))
            hits.append(
                RetrievalHit(
                    paper_id=paper_id,
                    compound_id=compound["compound_id"],
                    target_category=category,
                    source_id=sid,
                    retrieval_method=method[sid],
                    query=category,
                    score=scored[sid],
                    is_relevant=is_rel,
                    relevance_type=rel_type,
                )
            )
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Compound-conditioned semantic RAG.")
    ap.add_argument("--paper-ids", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--categories", type=str, default=None,
        help=f"Comma-separated subset of {ALL_CATEGORIES} (default: all three)",
    )
    args = ap.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    model = cfg["llm"]["relevance_model"]
    query_concepts = load_yaml(Path(__file__).resolve().parent.parent / "config" / "retrieval_queries.yaml")

    categories = ALL_CATEGORIES
    if args.categories:
        wanted_cats = {c.strip().lower() for c in args.categories.split(",")}
        unknown = wanted_cats - set(ALL_CATEGORIES)
        if unknown:
            raise SystemExit(f"unknown --categories {sorted(unknown)}; choose from {ALL_CATEGORIES}")
        categories = [c for c in ALL_CATEGORIES if c in wanted_cats]

    units_by_paper: dict[str, list[dict]] = defaultdict(list)
    for u in read_jsonl(data_dir / "chunks" / "02_chunks.jsonl"):
        units_by_paper[u["paper_id"]].append(u)

    compounds_by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in read_jsonl(data_dir / "registry" / "04_compound_registry.jsonl"):
        compounds_by_paper[c["paper_id"]].append(c)

    paper_ids = sorted(compounds_by_paper.keys())
    if args.paper_ids:
        wanted = {p.strip().upper() for p in args.paper_ids.split(",")}
        paper_ids = [p for p in paper_ids if p in wanted]

    client = LLMClient(provider=cfg["llm"]["provider"], dry_run=args.dry_run)
    all_hits: list[RetrievalHit] = []
    for paper_id in paper_ids:
        for compound in compounds_by_paper[paper_id]:
            logger.info("retrieving evidence for %s (categories=%s)", compound["compound_id"], categories)
            hits = retrieve_for_compound(client, model, cfg, query_concepts, paper_id, compound, units_by_paper[paper_id], categories)
            all_hits.extend(hits)
            n_rel = sum(1 for h in hits if h.is_relevant)
            logger.info("  -> %d candidates, %d marked relevant", len(hits), n_rel)

    write_jsonl(data_dir / "retrieval" / "05_retrieval_hits.jsonl", (h.model_dump() for h in all_hits))
    logger.info("wrote %d retrieval hits", len(all_hits))


if __name__ == "__main__":
    main()
