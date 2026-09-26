"""Schemas for retrieval results and evidence dossiers."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

TargetCategory = Literal["cation", "connectivity", "synthesis"]


class RetrievalHit(BaseModel):
    paper_id: str
    compound_id: str
    target_category: TargetCategory
    source_id: str
    retrieval_method: Literal["lexical_bm25", "dense", "alias_match", "neighbor", "counter_evidence"]
    query: str
    score: float
    is_relevant: Optional[bool] = None
    relevance_type: Optional[str] = None
    relevance_reason: Optional[str] = None


class EvidenceDossier(BaseModel):
    paper_id: str
    compound_id: str
    target_category: TargetCategory
    evidence_text: str
    source_ids: list[str] = Field(default_factory=list)
    not_reported: bool = Field(
        False, description="True when no explicit evidence was found for this category."
    )
    model: str
    prompt_version: str
    created_at: str
