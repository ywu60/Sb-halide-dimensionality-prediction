"""Extract cation and dimensionality fields and assemble compound records.

Synthesis evidence is copied from its source-tracked dossier without a
separate structured synthesis call.

Writes data/records/07_extracted_records.jsonl (one CompoundRecord per compound).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import CompoundRecord
from src.llm_client import LLMClient
from src.screen_compounds import select_sb_chunks
from src.utils import get_logger, load_config, load_prompt, load_yaml, read_jsonl, write_jsonl

logger = get_logger("extract_records")
PROMPT_VERSION = "v1"

CATION_SCHEMA = {
    "type": "object",
    "properties": {
        "cations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "abbreviation": {"type": ["string", "null"]},
                    "formula_explicit": {"type": ["string", "null"]},
                    "formula_status": {"type": "string", "enum": ["explicit", "not_reported", "ambiguous"]},
                },
                "required": ["name", "abbreviation", "formula_explicit", "formula_status"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cations"],
    "additionalProperties": False,
}

DIMENSIONALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "sb_oxidation_state_reported": {"type": ["string", "null"]},
        "halides_present": {"type": "array", "items": {"type": "string"}},
        "halides_bonded_to_sb": {"type": "array", "items": {"type": "string"}},
        "sb_halide_unit_reported": {"type": ["string", "null"]},
        "sb_halide_dimensionality_llm": {"type": "string", "enum": ["0D", "1D", "2D", "3D", "Unknown"]},
        "dimensionality_reasoning": {"type": "string"},
        "dimensionality_evidence_status": {"type": "string", "enum": ["sufficient", "insufficient", "conflicting"]},
        "dimensionality_review_reason": {"type": ["string", "null"]},
    },
    "required": [
        "sb_oxidation_state_reported",
        "halides_present",
        "halides_bonded_to_sb",
        "sb_halide_unit_reported",
        "sb_halide_dimensionality_llm",
        "dimensionality_reasoning",
        "dimensionality_evidence_status",
        "dimensionality_review_reason",
    ],
    "additionalProperties": False,
}


GLOSSARY_SCHEMA = {
    "type": "object",
    "properties": {
        "abbreviations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "definition": {"type": "string"},
                    "defined_in_evidence": {"type": "boolean"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symbol", "definition", "defined_in_evidence", "source_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["abbreviations"],
    "additionalProperties": False,
}


def _cation_stub(payload: dict) -> dict:
    return {"cations": []}


def _glossary_stub(payload: dict) -> dict:
    return {"abbreviations": []}


def _dim_stub(payload: dict) -> dict:
    return {
        "sb_oxidation_state_reported": None,
        "halides_present": [],
        "halides_bonded_to_sb": [],
        "sb_halide_unit_reported": None,
        "sb_halide_dimensionality_llm": "Unknown",
        "dimensionality_reasoning": "dry-run stub — no evidence evaluated",
        "dimensionality_evidence_status": "insufficient",
        "dimensionality_review_reason": "dry_run",
    }


def extract_cation(client: LLMClient, model: str, paper_id: str, compound_id: str, dossier: dict) -> dict:
    system_prompt = load_prompt("05_extract_cation.md")
    user_payload = {
        "task": "extract_cation",
        "paper_id": paper_id,
        "compound_id": compound_id,
        "instructions": "Extract structured cation fields from this evidence text only.",
        "evidence_units": [{"source_id": "dossier", "text": dossier["evidence_text"]}],
        "output_schema": CATION_SCHEMA,
    }
    result = client.call_structured(
        task="extract_cation",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=CATION_SCHEMA,
        schema_name="cation_extraction",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_cation_stub,
    )
    return result.data


def extract_formula_glossary(
    client: LLMClient,
    model: str,
    paper_id: str,
    compound_id: str,
    compound_formula: str | None,
    cation_formula: list,
    evidence_units: list[dict],
) -> list[dict]:
    if not compound_formula:
        return []
    system_prompt = load_prompt("08_formula_glossary.md")
    user_payload = {
        "task": "formula_glossary",
        "paper_id": paper_id,
        "compound_id": compound_id,
        "compound_formula_reported": compound_formula,
        "cation_formula_explicit": cation_formula,
        "instructions": "Identify and define every non-standard symbol used in the formula(s) above.",
        "evidence_units": [{"source_id": u["source_id"], "text": u["text"]} for u in evidence_units],
        "output_schema": GLOSSARY_SCHEMA,
    }
    result = client.call_structured(
        task="formula_glossary",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=GLOSSARY_SCHEMA,
        schema_name="formula_glossary",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_glossary_stub,
    )
    return result.data.get("abbreviations", [])


def format_glossary(entries: list[dict]) -> str | None:
    if not entries:
        return None
    parts = []
    for e in entries:
        if e.get("defined_in_evidence", True):
            parts.append(f"{e['symbol']} = {e['definition']}")
        else:
            parts.append(f"{e['symbol']} = [NOT DEFINED IN EVIDENCE — check source PDF]")
    return "; ".join(parts)


def reason_dimensionality(
    client: LLMClient, model: str, rules: dict, paper_id: str, compound_id: str, dossier: dict
) -> dict:
    base_prompt = load_prompt("06_reason_dimensionality.md")
    rules_block = (
        "\n\n## Reference rule definitions (authoritative)\n"
        f"{rules['definitions']}\n\nExcluded interactions: {rules['excluded_interactions']}\n"
        f"Procedure: {rules['reasoning_procedure']}"
    )
    system_prompt = base_prompt + rules_block
    user_payload = {
        "task": "reason_dimensionality",
        "paper_id": paper_id,
        "compound_id": compound_id,
        "instructions": "Apply the dimensionality definition strictly to this connectivity evidence text only.",
        "evidence_units": [{"source_id": "dossier", "text": dossier["evidence_text"]}],
        "output_schema": DIMENSIONALITY_SCHEMA,
    }
    result = client.call_structured(
        task="reason_dimensionality",
        system_prompt=system_prompt,
        user_payload=user_payload,
        json_schema=DIMENSIONALITY_SCHEMA,
        schema_name="dimensionality_reasoning",
        model=model,
        prompt_version=PROMPT_VERSION,
        stub_fn=_dim_stub,
    )
    return result.data


def assemble_record(
    paper_id: str,
    compound: dict,
    dossiers: dict[str, dict],
    cation_data: dict,
    dim_data: dict,
    glossary_entries: list[dict] | None = None,
) -> CompoundRecord:
    cations = cation_data.get("cations", [])
    cation_dossier = dossiers.get("cation", {})
    conn_dossier = dossiers.get("connectivity", {})
    syn_dossier = dossiers.get("synthesis", {})

    dim_label = dim_data["sb_halide_dimensionality_llm"]
    dim_status = dim_data["dimensionality_evidence_status"]
    if dim_status != "sufficient" and dim_label != "Unknown":
        logger.warning(
            "%s: forcing Unknown (model returned %s with evidence_status=%s)",
            compound["compound_id"], dim_label, dim_status,
        )
        dim_label = "Unknown"

    return CompoundRecord(
        paper_id=paper_id,
        compound_id=compound["compound_id"],
        label_in_paper=", ".join(compound.get("labels_in_paper", [])) or None,
        compound_name_reported=compound.get("compound_name_reported"),
        compound_formula_reported=compound.get("compound_formula_reported"),
        formula_abbreviation_glossary=format_glossary(glossary_entries or []),
        cation_name_reported=[c.get("name") for c in cations],
        cation_abbreviation_reported=[c.get("abbreviation") for c in cations],
        cation_formula_explicit=[c.get("formula_explicit") for c in cations],
        cation_formula_status=[c.get("formula_status", "not_reported") for c in cations],
        cation_evidence_text=cation_dossier.get("evidence_text", ""),
        cation_source=cation_dossier.get("source_ids", []),
        sb_oxidation_state_reported=dim_data.get("sb_oxidation_state_reported"),
        halides_present=dim_data.get("halides_present", []),
        halides_bonded_to_sb=dim_data.get("halides_bonded_to_sb", []),
        sb_halide_unit_reported=dim_data.get("sb_halide_unit_reported"),
        sb_halide_connectivity_evidence_text=conn_dossier.get("evidence_text", ""),
        connectivity_source=conn_dossier.get("source_ids", []),
        sb_halide_dimensionality_llm=dim_label,
        dimensionality_reasoning=dim_data.get("dimensionality_reasoning"),
        dimensionality_evidence_status=dim_status,
        dimensionality_review_reason=dim_data.get("dimensionality_review_reason"),
        synthesis_target_compound=compound.get("compound_name_reported") or compound["compound_id"],
        synthesis_evidence_text=syn_dossier.get("evidence_text", ""),
        synthesis_source=syn_dossier.get("source_ids", []),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract structured cation + dimensionality fields and assemble records.")
    ap.add_argument("--paper-ids", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    rules = load_yaml(Path(__file__).resolve().parent.parent / "config" / "dimensionality_rules.yaml")

    compounds_by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in read_jsonl(data_dir / "registry" / "04_compound_registry.jsonl"):
        compounds_by_paper[c["paper_id"]].append(c)

    units_by_paper: dict[str, list[dict]] = defaultdict(list)
    for u in read_jsonl(data_dir / "chunks" / "02_chunks.jsonl"):
        units_by_paper[u["paper_id"]].append(u)

    dossiers_by_compound: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for d in read_jsonl(data_dir / "dossiers" / "06_evidence_dossiers.jsonl"):
        dossiers_by_compound[(d["paper_id"], d["compound_id"])][d["target_category"]] = d

    paper_ids = sorted(compounds_by_paper.keys())
    if args.paper_ids:
        wanted = {p.strip().upper() for p in args.paper_ids.split(",")}
        paper_ids = [p for p in paper_ids if p in wanted]

    client = LLMClient(provider=cfg["llm"]["provider"], dry_run=args.dry_run)
    records: list[CompoundRecord] = []
    for paper_id in paper_ids:
        for compound in compounds_by_paper[paper_id]:
            cid = compound["compound_id"]
            dossiers = dossiers_by_compound.get((paper_id, cid), {})
            logger.info("extracting %s", cid)

            cation_data = extract_cation(client, cfg["llm"]["cation_model"], paper_id, cid, dossiers.get("cation", {"evidence_text": ""}))
            dim_data = reason_dimensionality(client, cfg["llm"]["dimensionality_model"], rules, paper_id, cid, dossiers.get("connectivity", {"evidence_text": ""}))
            glossary_entries = extract_formula_glossary(
                client, cfg["llm"]["cation_model"], paper_id, cid,
                compound.get("compound_formula_reported"),
                [c.get("formula_explicit") for c in cation_data.get("cations", [])],
                select_sb_chunks(units_by_paper[paper_id]),
            )
            record = assemble_record(paper_id, compound, dossiers, cation_data, dim_data, glossary_entries)
            records.append(record)
            logger.info("  -> dimensionality=%s (%s)", record.sb_halide_dimensionality_llm, record.dimensionality_evidence_status)

    write_jsonl(data_dir / "records" / "07_extracted_records.jsonl", (r.model_dump() for r in records))
    logger.info("wrote %d compound records", len(records))


if __name__ == "__main__":
    main()
