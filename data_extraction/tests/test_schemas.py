import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import CationRecord, DimensionalityReasoning, DocumentUnit


def _now():
    from src.utils import now_iso
    return now_iso()


def test_document_unit_roundtrip():
    u = DocumentUnit(
        source_id="P0001_Main_Page5_Paragraph3",
        paper_id="P0001",
        page=5,
        unit_type="paragraph",
        order_index=12,
        text="SbCl3 was dissolved in HCl.",
    )
    assert u.doc_type == "main"
    assert DocumentUnit(**u.model_dump()) == u


def test_cation_formula_status_not_reported_requires_null_formula():
    with pytest.raises(ValidationError):
        CationRecord(
            paper_id="P0001",
            compound_id="P0001_C1",
            cation_name_reported=["4-aminopyridinium"],
            cation_abbreviation_reported=["4AP"],
            cation_formula_explicit=["C5H7N2+"],
            cation_formula_status=["not_reported"],
            model="gpt-4.1",
            prompt_version="v1",
            created_at=_now(),
        )


def test_cation_record_allows_explicit_formula():
    r = CationRecord(
        paper_id="P0001",
        compound_id="P0001_C1",
        cation_name_reported=["4-aminopyridinium"],
        cation_abbreviation_reported=["4AP"],
        cation_formula_explicit=["C5H7N2+"],
        cation_formula_status=["explicit"],
        model="gpt-4.1",
        prompt_version="v1",
        created_at=_now(),
    )
    assert r.cation_formula_status == ["explicit"]


def test_dimensionality_unknown_required_when_insufficient():
    with pytest.raises(ValidationError):
        DimensionalityReasoning(
            paper_id="P0001",
            compound_id="P0001_C1",
            sb_halide_dimensionality_llm="1D",
            dimensionality_reasoning="incomplete evidence but guessed 1D",
            dimensionality_evidence_status="insufficient",
            model="gpt-4.1",
            prompt_version="v1",
            created_at=_now(),
        )


def test_dimensionality_unknown_ok_when_insufficient():
    d = DimensionalityReasoning(
        paper_id="P0001",
        compound_id="P0001_C1",
        sb_halide_dimensionality_llm="Unknown",
        dimensionality_reasoning="evidence does not establish bonded periodicity",
        dimensionality_evidence_status="insufficient",
        model="gpt-4.1",
        prompt_version="v1",
        created_at=_now(),
    )
    assert d.sb_halide_dimensionality_llm == "Unknown"
