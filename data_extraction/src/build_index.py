"""Per-paper lexical (BM25) index — plan §4.4, §6.3.

Retrieval is deliberately restricted to one paper (and its SI) at a time, so
the index is cheap to build on demand rather than persisted globally. Dense
retrieval is a pluggable extension point (see `DenseIndex` stub below) —
plan §6.3 names SentenceTransformers + FAISS but neither is required for the
pipeline to run end-to-end.
"""
from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, units: list[dict]):
        self.units = units
        self._tokenized = [tokenize(u["text"]) for u in units]
        self._bm25: Optional[BM25Okapi] = BM25Okapi(self._tokenized) if units else None

    def query(self, text: str, top_k: int) -> list[tuple[dict, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(text))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.units[i], float(scores[i])) for i in ranked if scores[i] > 0]


class DenseIndex:
    """Placeholder for a SentenceTransformers + FAISS dense index (plan §6.3).

    Not wired up in this scaffold — BM25 + alias matching already gives
    reasonable recall for a pilot. Swap in here without touching callers:
    `retrieve_evidence.py` only calls `.query(text, top_k)`.
    """

    def __init__(self, units: list[dict]):
        self.units = units

    def query(self, text: str, top_k: int) -> list[tuple[dict, float]]:
        return []
