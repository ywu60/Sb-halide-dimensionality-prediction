import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.validators import check_dimensionality_rules


def test_valid_0d_with_sufficient_evidence_passes():
    records = [{"compound_id": "P0099_C1", "sb_halide_dimensionality_llm": "0D", "dimensionality_evidence_status": "sufficient"}]
    assert check_dimensionality_rules(records) == []


def test_guessed_label_with_insufficient_evidence_flagged():
    records = [{"compound_id": "P0099_C1", "sb_halide_dimensionality_llm": "1D", "dimensionality_evidence_status": "insufficient"}]
    violations = check_dimensionality_rules(records)
    assert len(violations) == 1
    assert "must be 'Unknown'" in violations[0]


def test_unknown_with_conflicting_evidence_is_fine():
    records = [{"compound_id": "P0099_C1", "sb_halide_dimensionality_llm": "Unknown", "dimensionality_evidence_status": "conflicting"}]
    assert check_dimensionality_rules(records) == []


def test_invalid_label_flagged():
    records = [{"compound_id": "P0099_C1", "sb_halide_dimensionality_llm": "4D", "dimensionality_evidence_status": "sufficient"}]
    violations = check_dimensionality_rules(records)
    assert any("invalid dimensionality label" in v for v in violations)
