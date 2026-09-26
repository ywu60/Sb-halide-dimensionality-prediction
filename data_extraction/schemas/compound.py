"""Schemas for paper eligibility and the compound registry."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EligibilityDecision(BaseModel):
    paper_id: str
    candidate_label: str = Field(..., description="Label/name/formula as it appears in the paper")
    is_eligible: bool
    reason: str
    decision_category: Literal[
        "stoichiometric_sb_compound",
        "doped_or_trace_sb",
        "device_or_composite",
        "non_sb_precursor",
        "comparison_analogue",
        "background_only",
        "other_excluded",
    ]
    source_ids: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
    created_at: str


class CompoundRegistryEntry(BaseModel):
    paper_id: str
    compound_id: str = Field(..., description="Stable id, e.g. P0001_C1")
    labels_in_paper: list[str] = Field(default_factory=list)
    compound_name_reported: Optional[str] = None
    compound_formula_reported: Optional[str] = None
    supporting_source_ids: list[str] = Field(default_factory=list)
    model: str
    prompt_version: str
    created_at: str
