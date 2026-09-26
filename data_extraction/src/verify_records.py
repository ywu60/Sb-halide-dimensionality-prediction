"""Verify records, search for counter-evidence, and retry failed categories.

For each compound record:
  1. verify every field against its cited source passages;
  2. scan the paper for counter-evidence phrases not already covered by the
     cited sources;
  3. if verification fails, run a targeted re-retrieval for the failed
     category only (not the whole paper) and redo just that field, bounded
     by `verification.max_retrieval_cycles`;
  4. if still unresolved, leave the record as `incomplete_or_conflicting`
     with `Unknown` dimensionality where applicable — never guess.

Writes data/records/08_verified_records.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import CompoundRecord
from src.build_dossiers import build_dossier
from src.build_index import BM25Index
from src.extract_records import extract_cation, reason_dimensionality
from src.llm_client import LLMClient
from src.utils import get_logger, load_config, load_prompt, load_yaml, read_jsonl, write_jsonl
from src.validators import compute_automatic_flags

logger = get_logger("verify_records")
PROMPT_VERSION = "v1"

FIELDS_TO_CHECK = [
    "compound_identity",
    "cation_name",
    "cation_formula",
    "connectivity_evidence",
    "dimensionality_label",
    "synthesis_target",
    "row_sanity",
]
FIELD_TO_CATEGORY = {
    "cation_name": "cation",
    "cation_formula": "cation",
    "connectivity_evidence": "connectivity",
    "dimensionality_label": "connectivity",
    "synthesis_target": "synthesis",
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "field_checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string", "enum": FIELDS_TO_CHECK},
                    "status": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
                },
                "required": ["field_name", "status"],
                "additionalProperties": False,
            },
        },
        "counter_evidence_found": {"type": "boolean"},
        "counter_evidence_notes": {"type": ["string", "null"]},
        "recovery_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"field_name": {"type": "string", "enum": FIELDS_TO_CHECK}, "query": {"type": "string"}},
                "required": ["field_name", "query"],
                "additionalProperties": False,
            },
        },
        "overall_status": {"type": "string", "enum": ["supported", "incomplete_or_conflicting"]},
    },
    "required": ["field_checks", "counter_evidence_found", "counter_evidence_notes", "recovery_queries", "overall_status"],
    "additionalProperties": False,
}


def _stub(payload: dict) -> dict:
    return {
        "field_checks": [{"field_name": f, "status": "uncertain"} for f in FIELDS_TO_CHECK],
        "counter_evidence_found": False,
        "counter_evidence_notes": None,
        "recovery_queries": [],
        "overall_status": "incomplete_or_conflicting",
    }


def find_counter_evidence(units: list[dict], phrases: list[str], covered_source_ids: set[str]) -> list[str]:
    hits = []
    for u in units:
        if u["source_id"] in covered_source_ids:
            continue
        low = u["text"].lower()
        for p in phrases:
            if p in low:
                hits.append(f"{u['source_id']}: '{p}'")
                break
    return hits


def verify_once(
    client: LLMClient,
    model: str,
    record: dict,
    aliases: list[str],
    units_by_id: dict[str, dict],
    counter_evidence_phrases: list[str],
    cycle: int,
) -> dict:
    cited_ids = set(record["cation_source"]) | set(record["connectivity_source"]) | set(record["synthesis_source"])
    cited_units = [units_by_id[sid] for sid in cited_ids if sid in units_by_id]
    counter_hits = find_counter_evidence(list(units_by_id.values()), counter_evidence_phrases, cited_ids)

    system_prompt = load_prompt("07_verify_record.md")
    user_payload = {
        "task": "verify_record",
        "paper_id": record["paper_id"],
        "compound_id": record["compound_id"],
        "compound_aliases": aliases,
        "fields_to_check": FIELDS_TO_CHECK,
        "draft_record": {
            "compound_name_reported": record["compound_name_reported"],
            "compound_formula_reported": record["compound_formula_reported"],
            "cation_name_reported": record["cation_name_reported"],
            "cation_formula_explicit": record["cation_formula_explicit"],
            "cation_formula_status": record["cation_formula_status"],
            "sb_halide_dimensionality_llm": record["sb_halide_dimensionality_llm"],
            "dimensionality_evidence_status": record["dimensionality_evidence_status"],
            "synthesis_target_compound": record["synthesis_target_compound"],
        },
        "cited_evidence_units": [{"source_id": u["source_id"], "text": u["text"]} for u in cited_units],
        "possible_counter_evidence_elsewhere_in_paper": counter_hits[:15],
        "output_schema": VERIFY_SCHEMA,
    }
    result = client.call_structured(
        task="verify_record",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=VERIFY_SCHEMA,
        schema_name="verification_result",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_stub,
    )
    d = result.data
    return {
        "field_checks": {f["field_name"]: f["status"] for f in d["field_checks"]},
        "failed_fields": [f["field_name"] for f in d["field_checks"] if f["status"] != "pass"],
        "counter_evidence_found": d["counter_evidence_found"],
        "counter_evidence_notes": d["counter_evidence_notes"],
        "recovery_queries": {r["field_name"]: r["query"] for r in d["recovery_queries"]},
        "overall_status": d["overall_status"],
        "retrieval_cycle": cycle,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "created_at": result.created_at,
    }


def retry_category(
    client: LLMClient,
    cfg: dict,
    paper_id: str,
    compound: dict,
    record: dict,
    category: str,
    query: str,
    units: list[dict],
    units_by_id: dict[str, dict],
    dossiers: dict[str, dict],
) -> None:
    index = BM25Index(units)
    top_k = cfg["retrieval"]["bm25_top_k"]
    new_hits = index.query(query, top_k)
    existing_source_ids = set(dossiers.get(category, {}).get("source_ids", []))
    new_units = [u for u, score in new_hits if u["source_id"] not in existing_source_ids][:10]
    if not new_units:
        logger.info("  retry(%s): no new candidates for query '%s'", category, query)
        return

    prior_units = [units_by_id[sid] for sid in existing_source_ids if sid in units_by_id]
    combined_units = prior_units + new_units
    aliases = [compound.get("compound_formula_reported"), compound.get("compound_name_reported")] + compound.get("labels_in_paper", [])
    aliases = [a for a in aliases if a]
    new_dossier = build_dossier(client, cfg["llm"]["dossier_model"], paper_id, compound["compound_id"], category, combined_units, aliases)
    dossiers[category] = new_dossier.model_dump()

    if category == "cation":
        cation_data = extract_cation(client, cfg["llm"]["cation_model"], paper_id, compound["compound_id"], dossiers[category])
        cations = cation_data.get("cations", [])
        record["cation_name_reported"] = [c.get("name") for c in cations]
        record["cation_abbreviation_reported"] = [c.get("abbreviation") for c in cations]
        record["cation_formula_explicit"] = [c.get("formula_explicit") for c in cations]
        record["cation_formula_status"] = [c.get("formula_status", "not_reported") for c in cations]
        record["cation_evidence_text"] = dossiers[category]["evidence_text"]
        record["cation_source"] = dossiers[category]["source_ids"]
    elif category == "connectivity":
        rules = load_yaml(Path(__file__).resolve().parent.parent / "config" / "dimensionality_rules.yaml")
        dim_data = reason_dimensionality(client, cfg["llm"]["dimensionality_model"], rules, paper_id, compound["compound_id"], dossiers[category])
        dim_label = dim_data["sb_halide_dimensionality_llm"]
        if dim_data["dimensionality_evidence_status"] != "sufficient" and dim_label != "Unknown":
            dim_label = "Unknown"
        record["sb_oxidation_state_reported"] = dim_data.get("sb_oxidation_state_reported")
        record["halides_present"] = dim_data.get("halides_present", [])
        record["halides_bonded_to_sb"] = dim_data.get("halides_bonded_to_sb", [])
        record["sb_halide_unit_reported"] = dim_data.get("sb_halide_unit_reported")
        record["sb_halide_connectivity_evidence_text"] = dossiers[category]["evidence_text"]
        record["connectivity_source"] = dossiers[category]["source_ids"]
        record["sb_halide_dimensionality_llm"] = dim_label
        record["dimensionality_reasoning"] = dim_data.get("dimensionality_reasoning")
        record["dimensionality_evidence_status"] = dim_data["dimensionality_evidence_status"]
        record["dimensionality_review_reason"] = dim_data.get("dimensionality_review_reason")
    elif category == "synthesis":
        record["synthesis_evidence_text"] = dossiers[category]["evidence_text"]
        record["synthesis_source"] = dossiers[category]["source_ids"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify records with bounded targeted re-retrieval.")
    ap.add_argument("--paper-ids", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    model = cfg["llm"]["verification_model"]
    max_cycles = cfg["verification"]["max_retrieval_cycles"]
    query_cfg = load_yaml(Path(__file__).resolve().parent.parent / "config" / "retrieval_queries.yaml")
    counter_phrases = [p.lower() for p in query_cfg["counter_evidence"]]

    units_by_paper: dict[str, list[dict]] = defaultdict(list)
    for u in read_jsonl(data_dir / "chunks" / "02_chunks.jsonl"):
        units_by_paper[u["paper_id"]].append(u)

    compounds_by_key: dict[tuple[str, str], dict] = {}
    for c in read_jsonl(data_dir / "registry" / "04_compound_registry.jsonl"):
        compounds_by_key[(c["paper_id"], c["compound_id"])] = c

    dossiers_by_compound: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for d in read_jsonl(data_dir / "dossiers" / "06_evidence_dossiers.jsonl"):
        dossiers_by_compound[(d["paper_id"], d["compound_id"])][d["target_category"]] = d

    records = list(read_jsonl(data_dir / "records" / "07_extracted_records.jsonl"))
    paper_ids = sorted({r["paper_id"] for r in records})
    if args.paper_ids:
        wanted = {p.strip().upper() for p in args.paper_ids.split(",")}
        paper_ids = [p for p in paper_ids if p in wanted]

    client = LLMClient(provider=cfg["llm"]["provider"], dry_run=args.dry_run)
    verified: list[CompoundRecord] = []
    for record in records:
        if record["paper_id"] not in paper_ids:
            continue
        paper_id, cid = record["paper_id"], record["compound_id"]
        units = units_by_paper[paper_id]
        units_by_id = {u["source_id"]: u for u in units}
        compound = compounds_by_key.get((paper_id, cid), {"compound_id": cid, "labels_in_paper": []})
        dossiers = dossiers_by_compound.get((paper_id, cid), {})
        aliases = [compound.get("compound_formula_reported"), compound.get("compound_name_reported")] + compound.get("labels_in_paper", [])
        aliases = [a for a in aliases if a]

        logger.info("verifying %s", cid)
        cycle = 0
        v = verify_once(client, model, record, aliases, units_by_id, counter_phrases, cycle)
        while v["overall_status"] != "supported" and cycle < max_cycles:
            categories_to_retry = {
                FIELD_TO_CATEGORY[f] for f in v["failed_fields"] if f in FIELD_TO_CATEGORY and f in v["recovery_queries"]
            }
            if not categories_to_retry:
                break
            cycle += 1
            for category in categories_to_retry:
                query = next(v["recovery_queries"][f] for f in v["failed_fields"] if FIELD_TO_CATEGORY.get(f) == category)
                logger.info("  cycle %d: re-retrieving '%s' for %s", cycle, category, cid)
                retry_category(client, cfg, paper_id, compound, record, category, query, units, units_by_id, dossiers)
            v = verify_once(client, model, record, aliases, units_by_id, counter_phrases, cycle)

        record["verification_status"] = v["overall_status"]
        record["verification_notes"] = v.get("counter_evidence_notes")
        record["retrieval_cycles_used"] = cycle
        if v["overall_status"] != "supported":
            logger.warning("  %s still %s after %d cycle(s)", cid, v["overall_status"], cycle)
            if record["sb_halide_dimensionality_llm"] != "Unknown" and "dimensionality_label" in v["failed_fields"]:
                record["sb_halide_dimensionality_llm"] = "Unknown"
                record["dimensionality_review_reason"] = "unresolved after verification: " + str(v.get("counter_evidence_notes"))

        flags = compute_automatic_flags(record)
        if v["field_checks"].get("row_sanity") in ("fail", "uncertain"):
            flags.append(f"LLM row_sanity check: {v['field_checks']['row_sanity']} — {v.get('counter_evidence_notes') or 'see verification_notes'}")
        if v["field_checks"].get("compound_identity") in ("fail", "uncertain"):
            flags.append(f"LLM compound_identity check: {v['field_checks']['compound_identity']}")
        record["automatic_flags"] = flags
        if flags:
            logger.info("  %s: %d automatic flag(s)", cid, len(flags))

        verified.append(CompoundRecord(**record))

    write_jsonl(data_dir / "records" / "08_verified_records.jsonl", (r.model_dump() for r in verified))
    logger.info("wrote %d verified records", len(verified))


if __name__ == "__main__":
    main()
