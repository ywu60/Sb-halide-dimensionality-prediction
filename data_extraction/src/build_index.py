"""Build an in-memory BM25 index for each paper and its supporting files.

The index is built on demand because retrieval is restricted to one paper at
a time. `DenseIndex` defines an optional interface for another retrieval
backend without adding dependencies to the default pipeline.
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
    def __init__(self, units: list[dict]):
        self.units = units

    def query(self, text: str, top_k: int) -> list[tuple[dict, float]]:
        return []
