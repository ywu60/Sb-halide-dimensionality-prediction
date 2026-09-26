"""Run deterministic checks over assembled compound records."""
from __future__ import annotations

ALLOWED_DIMENSIONALITY = {"0D", "1D", "2D", "3D", "Unknown"}
HALIDE_SYMBOLS = ("F", "Cl", "Br", "I")
UNRECOVERABLE_MARKER = "[?]"

_ALLOWED_EXTRA_CHARS = set("·°±→↔≡′″ÅåΔδαβγμωΩ–—''\"\"…‹›⋯×−⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹")


def scan_text_sanity(value: str) -> list[str]:
    bad = []
    for ch in value:
        if ch in ("\n", "\t"):
            continue
        cp = ord(ch)
        if 0x20 <= cp <= 0x7E:
            continue
        if ch in _ALLOWED_EXTRA_CHARS:
            continue
        bad.append(ch)
    seen = []
    for ch in bad:
        if ch not in seen:
            seen.append(ch)
    return seen


def check_record_text_sanity(record: dict, fields: list[str]) -> list[str]:
    flags = []
    for field in fields:
        val = record.get(field)
        if isinstance(val, str) and val:
            flags.extend(_text_sanity_flags_for(field, val))
    return flags


def check_halide_consistency(record: dict) -> list[str]:
    formula = record.get("compound_formula_reported") or ""
    halides_bonded = set(record.get("halides_bonded_to_sb") or [])
    present_in_formula = {h for h in HALIDE_SYMBOLS if h in formula}
    flags = []
    missing = present_in_formula - halides_bonded
    extra = halides_bonded - present_in_formula
    if missing:
        flags.append(
            f"halide(s) {sorted(missing)} appear in compound_formula_reported "
            f"('{formula}') but are not listed in halides_bonded_to_sb"
        )
    if extra:
        flags.append(
            f"halides_bonded_to_sb lists {sorted(extra)}, not found in "
            f"compound_formula_reported ('{formula}')"
        )
    return flags


def _text_sanity_flags_for(label: str, value: str) -> list[str]:
    flags = []
    if UNRECOVERABLE_MARKER in value:
        flags.append(
            f"{label}: contains '{UNRECOVERABLE_MARKER}' — a character (likely a Greek letter or "
            "symbol, e.g. bridging-ligand 'μ') could not be recovered from the model's output and "
            "was masked rather than silently dropped. Check the source PDF for the true value."
        )
    bad = scan_text_sanity(value)
    if bad:
        shown = ", ".join(f"U+{ord(c):04X} '{c}'" for c in bad[:5])
        flags.append(f"{label}: unexpected character(s) {shown} — verify against source PDF")
    return flags


def compute_automatic_flags(record: dict) -> list[str]:
    text_fields = [
        "compound_name_reported",
        "compound_formula_reported",
        "cation_formula_explicit",
        "sb_halide_unit_reported",
    ]
    flags = []
    for field in text_fields:
        val = record.get(field)
        if isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, str) and item:
                    flags.extend(_text_sanity_flags_for(f"{field}[{i}]", item))
        elif isinstance(val, str) and val:
            flags.extend(_text_sanity_flags_for(field, val))
    flags.extend(check_halide_consistency(record))
    return flags


def check_source_ids_exist(records: list[dict], valid_source_ids: set[str]) -> list[str]:
    violations = []
    for r in records:
        cited = set(r.get("cation_source", [])) | set(r.get("connectivity_source", [])) | set(r.get("synthesis_source", []))
        missing = cited - valid_source_ids
        for sid in missing:
            violations.append(f"{r['compound_id']}: cited source_id '{sid}' not found in chunk index")
    return violations


def check_compound_ids_in_registry(records: list[dict], registry_compound_ids: set[str]) -> list[str]:
    violations = []
    for r in records:
        if r["compound_id"] not in registry_compound_ids:
            violations.append(f"{r['compound_id']}: not present in compound registry")
    return violations


def check_dimensionality_rules(records: list[dict]) -> list[str]:
    violations = []
    for r in records:
        label = r.get("sb_halide_dimensionality_llm")
        status = r.get("dimensionality_evidence_status")
        if label not in ALLOWED_DIMENSIONALITY:
            violations.append(f"{r['compound_id']}: invalid dimensionality label '{label}'")
        if status != "sufficient" and label not in (None, "Unknown"):
            violations.append(
                f"{r['compound_id']}: dimensionality '{label}' with evidence_status '{status}' "
                "must be 'Unknown'"
            )
    return violations


def check_cation_formula_status(records: list[dict]) -> list[str]:
    violations = []
    for r in records:
        formulas = r.get("cation_formula_explicit", [])
        statuses = r.get("cation_formula_status", [])
        for formula, status in zip(formulas, statuses):
            if status == "not_reported" and formula is not None:
                violations.append(
                    f"{r['compound_id']}: cation_formula_explicit must be null when status is 'not_reported'"
                )
    return violations
