import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm_client import sanitize_json, sanitize_text
from src.validators import check_halide_consistency, compute_automatic_flags, scan_text_sanity


def test_sanitize_text_repairs_del_to_middle_dot():
    cleaned, changed = sanitize_text("[SbCl2(L)]\x7f(0.5thf)\x7f(0.5Et2O)")
    assert changed
    assert cleaned == "[SbCl2(L)]·(0.5thf)·(0.5Et2O)"


def test_sanitize_text_leaves_clean_text_alone():
    cleaned, changed = sanitize_text("[SbCl6]3-")
    assert not changed
    assert cleaned == "[SbCl6]3-"


def test_sanitize_json_recurses_into_nested_structures():
    out = sanitize_json({"formula": "A\x7fB", "list": ["C\x01D"]})
    assert out == {"formula": "A·B", "list": ["C[?]D"]}


def test_sanitize_text_repairs_truncated_mu_escape():
    cleaned, changed = sanitize_text("[{SbI(L)(\x03bc-I)}2]")
    assert changed
    assert cleaned == "[{SbI(L)(μ-I)}2]"


def test_sanitize_text_marks_unrecoverable_control_char_visibly_not_silently():
    cleaned, changed = sanitize_text("[{SbI(L)(\x03-I)}2]")
    assert changed
    assert "[?]" in cleaned
    assert cleaned == "[{SbI(L)([?]-I)}2]"


def test_scan_text_sanity_flags_lozenge_but_not_middle_dot():
    assert scan_text_sanity("[SbCl2(L)]·(0.5thf)") == []
    assert scan_text_sanity("[Et2O◊◊◊H(L)H◊◊◊OEt2]") == ["◊"]


def test_halide_consistency_flags_mismatch():
    record = {"compound_formula_reported": "[SbCl2(L)]", "halides_bonded_to_sb": ["Br"]}
    flags = check_halide_consistency(record)
    assert len(flags) == 2  # Cl missing, Br extra
    assert any("Cl" in f for f in flags)
    assert any("Br" in f for f in flags)


def test_halide_consistency_passes_when_matching():
    record = {"compound_formula_reported": "[SbCl2(L)]", "halides_bonded_to_sb": ["Cl"]}
    assert check_halide_consistency(record) == []


def test_compute_automatic_flags_catches_control_char_in_formula():
    record = {
        "compound_name_reported": "test",
        "compound_formula_reported": "[SbCl2(L)]\x7f(0.5thf)",
        "cation_formula_explicit": [],
        "sb_halide_unit_reported": None,
        "halides_bonded_to_sb": ["Cl"],
    }
    flags = compute_automatic_flags(record)
    assert any("compound_formula_reported" in f for f in flags)
