"""Small shared helpers: I/O, hashing, timestamps, config loading.

Kept dependency-free (stdlib + yaml) so every stage script can import it
without pulling in the LLM client.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover — dotenv is a thin convenience, not a hard dependency
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path = REPO_ROOT / "config" / "pipeline.yaml") -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    papers_dir = Path(cfg["paths"]["papers_dir"])
    if not papers_dir.is_absolute():
        papers_dir = (REPO_ROOT / papers_dir).resolve()
    cfg["paths"]["papers_dir"] = str(papers_dir)
    data_dir = Path(cfg["paths"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    cfg["paths"]["data_dir"] = str(data_dir)
    return cfg


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict], mode: str = "w") -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, mode) as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    write_jsonl(path, [record], mode="a")


def load_prompt(name: str) -> str:
    p = REPO_ROOT / "prompts" / name
    return p.read_text()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger
