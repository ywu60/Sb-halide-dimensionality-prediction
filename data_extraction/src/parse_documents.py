"""Stage 1: layout-aware document parsing — plan §4.2.

Uses PyMuPDF for page-level text blocks, reading order, and native table
detection (`page.find_tables`). Docling is the recommended upgrade path for
production-grade layout parsing (plan §6.3, §13) but is not installed here;
this parser is the practical PyMuPDF fallback the plan explicitly allows.

Writes:
  data/parsed/01_documents.jsonl   (one row per paper — PaperMeta)
  data/chunks/02_chunks.jsonl      (one row per evidence unit — DocumentUnit)

The original block/row text is always retained verbatim in `text`; no
cleaning step replaces it (plan §4.2).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import DocumentUnit, PaperMeta  # noqa: E402
from src.utils import get_logger, load_config, now_iso, write_jsonl  # noqa: E402

logger = get_logger("parse_documents")
PARSER_VERSION = "pymupdf-1.27+heuristic-v1"
PAPER_ID_RE = re.compile(r"(P\d{3,5})", re.IGNORECASE)


def discover_papers(papers_dir: Path) -> dict[str, Path]:
    """Map paper_id -> pdf path. Skips duplicate-suffixed re-exports by
    preferring the shortest filename per paper_id (the plain 'P0001.pdf'
    form over 'P0001 02.02.30 02.02.30.pdf')."""
    found: dict[str, Path] = {}
    for pdf_path in papers_dir.rglob("*.pdf"):
        m = PAPER_ID_RE.search(pdf_path.stem)
        if not m:
            continue
        paper_id = m.group(1).upper()
        if paper_id not in found or len(pdf_path.name) < len(found[paper_id].name):
            found[paper_id] = pdf_path
    return found


def _bbox_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _classify(text: str) -> str:
    t = text.strip()
    if re.match(r"^(Figure|Fig\.|FIGURE)\s*\d", t):
        return "figure_caption"
    if len(t) < 70 and not t.endswith((".", ",", ";")) and t == t.strip():
        # short, unpunctuated line — plausible section heading
        letters = [c for c in t if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
            return "heading"
    return "paragraph"


def parse_pdf(paper_id: str, pdf_path: Path, min_paragraph_chars: int) -> tuple[PaperMeta, list[DocumentUnit]]:
    doc = fitz.open(pdf_path)
    units: list[DocumentUnit] = []
    current_section: str | None = None
    order_index = 0

    for page_no in range(len(doc)):
        page = doc[page_no]
        page_num_1based = page_no + 1
        counters: dict[str, int] = {}

        table_bboxes: list[tuple[float, float, float, float]] = []
        try:
            tables = page.find_tables()
        except Exception as exc:  # pragma: no cover — some pages have no tables
            logger.debug("find_tables failed on %s page %d: %s", paper_id, page_num_1based, exc)
            tables = []

        for t_idx, table in enumerate(tables, start=1):
            table_bboxes.append(tuple(table.bbox))
            try:
                rows = table.extract()
            except Exception:
                rows = []
            for r_idx, row in enumerate(rows, start=1):
                row_text = " | ".join(c.strip() if c else "" for c in row)
                if len(row_text.strip()) < 1:
                    continue
                counters["Table"] = t_idx
                source_id = f"{paper_id}_Main_Page{page_num_1based}_Table{t_idx}_Row{r_idx}"
                units.append(
                    DocumentUnit(
                        source_id=source_id,
                        paper_id=paper_id,
                        doc_type="main",
                        page=page_num_1based,
                        unit_type="table_row",
                        section=current_section,
                        order_index=order_index,
                        text=row_text,
                    )
                )
                order_index += 1

        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))  # top-to-bottom, left-to-right

        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            if not text or not text.strip():
                continue
            if any(_bbox_overlaps((x0, y0, x1, y1), tb) for tb in table_bboxes):
                continue
            for para in re.split(r"\n\s*\n", text):
                para = para.strip().replace("\n", " ")
                para = re.sub(r"\s{2,}", " ", para)
                if len(para) < min_paragraph_chars:
                    continue
                unit_type = _classify(para)
                if unit_type == "heading":
                    current_section = para
                label = {"paragraph": "Paragraph", "figure_caption": "FigureCaption", "heading": "Heading"}[unit_type]
                counters[label] = counters.get(label, 0) + 1
                source_id = f"{paper_id}_Main_Page{page_num_1based}_{label}{counters[label]}"
                units.append(
                    DocumentUnit(
                        source_id=source_id,
                        paper_id=paper_id,
                        doc_type="main",
                        page=page_num_1based,
                        unit_type=unit_type,
                        section=current_section,
                        order_index=order_index,
                        text=para,
                        bbox=[round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    )
                )
                order_index += 1

    for i, u in enumerate(units):
        if i > 0:
            u.prev_source_id = units[i - 1].source_id
        if i < len(units) - 1:
            u.next_source_id = units[i + 1].source_id

    meta = PaperMeta(
        paper_id=paper_id,
        main_path=str(pdf_path),
        si_path=None,
        n_pages_main=len(doc),
        n_pages_si=None,
        parsed_at=now_iso(),
        parser_version=PARSER_VERSION,
    )
    doc.close()
    return meta, units


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PDFs into layout-preserving evidence units.")
    ap.add_argument("--paper-ids", type=str, default=None, help="Comma-separated paper IDs, e.g. P0001,P0002")
    ap.add_argument("--limit", type=int, default=None, help="Only parse the first N discovered papers")
    args = ap.parse_args()

    cfg = load_config()
    papers_dir = Path(cfg["paths"]["papers_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    min_chars = cfg["parsing"]["min_paragraph_chars"]

    all_papers = discover_papers(papers_dir)
    logger.info("discovered %d papers under %s", len(all_papers), papers_dir)

    if args.paper_ids:
        wanted = {p.strip().upper() for p in args.paper_ids.split(",")}
        selected = {k: v for k, v in all_papers.items() if k in wanted}
        missing = wanted - selected.keys()
        if missing:
            logger.warning("requested paper IDs not found: %s", sorted(missing))
    elif args.limit:
        selected = dict(list(sorted(all_papers.items()))[: args.limit])
    else:
        selected = all_papers

    all_meta: list[PaperMeta] = []
    all_units: list[DocumentUnit] = []
    for paper_id, pdf_path in sorted(selected.items()):
        logger.info("parsing %s (%s)", paper_id, pdf_path.name)
        meta, units = parse_pdf(paper_id, pdf_path, min_chars)
        all_meta.append(meta)
        all_units.extend(units)
        logger.info("  -> %d pages, %d evidence units", meta.n_pages_main, len(units))

    write_jsonl(data_dir / "parsed" / "01_documents.jsonl", (m.model_dump() for m in all_meta))
    write_jsonl(data_dir / "chunks" / "02_chunks.jsonl", (u.model_dump() for u in all_units))
    logger.info("wrote %d papers, %d evidence units", len(all_meta), len(all_units))


if __name__ == "__main__":
    main()
