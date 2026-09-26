"""Document and evidence-unit schemas — pipeline plan §4.1-4.2."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PaperMeta(BaseModel):
    """One row per paper — 01_documents.jsonl."""

    paper_id: str = Field(..., description="Stable paper identifier, e.g. P0001")
    doi: Optional[str] = None
    title: Optional[str] = None
    main_path: str
    si_path: Optional[str] = None
    n_pages_main: int
    n_pages_si: Optional[int] = None
    parsed_at: str
    parser_version: str


class DocumentUnit(BaseModel):
    """A single layout-preserving evidence unit — 02_chunks.jsonl.

    Source IDs are stable and human-auditable, e.g.
    'P0001_Main_Page5_Paragraph3' or 'P0001_SI_Page12_Table2_Row4'.
    """

    source_id: str
    paper_id: str
    doc_type: Literal["main", "si"] = "main"
    page: int
    unit_type: Literal["paragraph", "table_row", "figure_caption", "heading"] = "paragraph"
    section: Optional[str] = None
    order_index: int = Field(..., description="Reading-order position within the document")
    text: str
    text_clean: Optional[str] = Field(
        None, description="Normalized text for retrieval only; never replaces `text`."
    )
    bbox: Optional[list[float]] = None
    prev_source_id: Optional[str] = None
    next_source_id: Optional[str] = None
