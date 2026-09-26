"""Structured-extraction, reasoning, verification, and record schemas.

Pipeline plan §4.6 (cation), §4.7 (dimensionality), §4.8 (record assembly),
§4.9 (verification), followed by direct dataset export.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

CationFormulaStatus = Literal["explicit", "not_reported", "ambiguous"]
Dimensionality = Literal["0D", "1D", "2D", "3D", "Unknown"]
EvidenceStatus = Literal["sufficient", "insufficient", "conflicting"]


class CationRecord(BaseModel):
    """§4.6 structured cation fields. Supports multiple cations per compound."""

    paper_id: str
    compound_id: str
    cation_name_reported: list[Optional[str]] = Field(default_factory=list)
    cation_abbreviation_reported: list[Optional[str]] = Field(default_factory=list)
    cation_formula_explicit: list[Optional[str]] = Field(default_factory=list)
    cation_formula_status: list[CationFormulaStatus] = Field(default_factory=list)
    cation_evidence_text: str = ""
    cation_source: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
    created_at: str

    @model_validator(mode="after")
    def _formula_status_consistency(self) -> "CationRecord":
        for formula, status in zip(self.cation_formula_explicit, self.cation_formula_status):
            if status == "not_reported" and formula is not None:
                raise ValueError(
                    "cation_formula_explicit must be null when cation_formula_status is "
                    "'not_reported' (plan §10.1)"
                )
        return self


class DimensionalityReasoning(BaseModel):
    """§4.7 output of the dimensionality-reasoning stage."""

    paper_id: str
    compound_id: str
    sb_oxidation_state_reported: Optional[str] = None
    halides_present: list[str] = Field(default_factory=list)
    halides_bonded_to_sb: list[str] = Field(default_factory=list)
    sb_halide_unit_reported: Optional[str] = None
    sb_halide_connectivity_evidence_text: str = ""
    connectivity_source: list[str] = Field(default_factory=list)
    sb_halide_dimensionality_llm: Dimensionality
    dimensionality_reasoning: str
    dimensionality_evidence_status: EvidenceStatus
    dimensionality_review_reason: Optional[str] = None
    model: str
    prompt_version: str
    created_at: str

    @model_validator(mode="after")
    def _unknown_requires_weak_evidence(self) -> "DimensionalityReasoning":
        if self.dimensionality_evidence_status != "sufficient" and self.sb_halide_dimensionality_llm != "Unknown":
            raise ValueError(
                "sb_halide_dimensionality_llm must be 'Unknown' when "
                "dimensionality_evidence_status is not 'sufficient' (plan §10.1)"
            )
        return self


class VerificationResult(BaseModel):
    """§4.9 verifier output for one compound record."""

    paper_id: str
    compound_id: str
    field_checks: dict[str, Literal["pass", "fail", "uncertain"]] = Field(default_factory=dict)
    failed_fields: list[str] = Field(default_factory=list)
    counter_evidence_found: bool = False
    counter_evidence_notes: Optional[str] = None
    recovery_queries: dict[str, str] = Field(
        default_factory=dict, description="target field -> targeted retrieval query"
    )
    overall_status: Literal["supported", "incomplete_or_conflicting"]
    retrieval_cycle: int = 0
    model: str
    prompt_version: str
    created_at: str


class CompoundRecord(BaseModel):
    """§4.8 fully assembled compound-level record — 07/08 stage output."""

    # identification
    paper_id: str
    compound_id: str
    label_in_paper: Optional[str] = None
    compound_name_reported: Optional[str] = None
    compound_formula_reported: Optional[str] = None
    formula_abbreviation_glossary: Optional[str] = Field(
        None, description="Definitions of non-standard symbols used in the reported formula(s), e.g. 'L = ...'"
    )

    # cation
    cation_name_reported: list[Optional[str]] = Field(default_factory=list)
    cation_abbreviation_reported: list[Optional[str]] = Field(default_factory=list)
    cation_formula_explicit: list[Optional[str]] = Field(default_factory=list)
    cation_formula_status: list[CationFormulaStatus] = Field(default_factory=list)
    cation_evidence_text: str = ""
    cation_source: list[str] = Field(default_factory=list)

    # connectivity evidence
    sb_oxidation_state_reported: Optional[str] = None
    halides_present: list[str] = Field(default_factory=list)
    halides_bonded_to_sb: list[str] = Field(default_factory=list)
    sb_halide_unit_reported: Optional[str] = None
    sb_halide_connectivity_evidence_text: str = ""
    connectivity_source: list[str] = Field(default_factory=list)

    # dimensionality reasoning
    sb_halide_dimensionality_llm: Dimensionality = "Unknown"
    dimensionality_reasoning: Optional[str] = None
    dimensionality_evidence_status: EvidenceStatus = "insufficient"
    dimensionality_review_reason: Optional[str] = None

    # synthesis
    synthesis_target_compound: Optional[str] = None
    synthesis_evidence_text: str = ""
    synthesis_source: list[str] = Field(default_factory=list)

    # verification
    verification_status: Optional[Literal["supported", "incomplete_or_conflicting"]] = None
    verification_notes: Optional[str] = None
    automatic_flags: list[str] = Field(
        default_factory=list, description="Deterministic + LLM sanity-check flags (§10.1-style checks)"
    )
    retrieval_cycles_used: int = 0

    # provenance
    pipeline_version: Optional[str] = None
    updated_at: Optional[str] = None
