"""Provider wrapper for structured, source-grounded LLM calls.

Design principles (plan §6.1, §8.1):
  - every call is logged with model id, prompt version, and timestamp;
  - low temperature, explicit JSON schema, bounded retry on invalid JSON;
  - responses are cached on (model, prompt_version, input hash) so repeated
    pipeline runs during development do not re-spend tokens;
  - a `dry_run` mode lets every stage run end-to-end with no API key and no
    cost, using a caller-supplied stub — useful for wiring tests and CI.

Only OpenAI is wired for now (per current project decision). The call
surface (`LLMClient.call_structured`) is intentionally provider-agnostic so
an Anthropic backend can be dropped in later without touching stage code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .utils import REPO_ROOT, ensure_dir, get_logger, now_iso, sha256_of

logger = get_logger("llm_client")

CACHE_DIR = REPO_ROOT / "data" / ".cache"

# Control characters observed in structured-output responses standing in
# for a punctuation/symbol character the model clearly intended (e.g.
# gpt-4.1-mini has been seen emitting U+007F DEL in place of the middle dot
# '·' used in solvate formulas like "[SbCl2(L)]·(0.5thf)", and "\x03bc" in
# place of Greek mu 'μ' used in bridging-ligand notation like "(μ-I)").
# Multi-character patterns are matched first (longest match wins) so "\x03bc"
# is repaired to 'μ' rather than treating the lone \x03 as an unknown byte.
_CONTROL_CHAR_REPAIRS = {
    "\x03bc": "μ",  # observed truncated μ (Greek mu) escape
    "\x7f": "·",    # DEL -> middle dot
}
_STRIP_CONTROL_CHARS = {chr(c) for c in list(range(0x00, 0x09)) + list(range(0x0B, 0x20))}
_UNRECOVERABLE_MARKER = "[?]"


def sanitize_text(value: str) -> tuple[str, bool]:
    """Returns (cleaned_value, was_modified). Known bad patterns are
    repaired to their almost-certain intended character. Any *other*
    control character is replaced with a visible "[?]" marker rather than
    silently deleted — losing a bridging-ligand symbol like 'μ' by deleting
    it outright previously turned "(μ-I)" into "(-I)" with no trace that
    anything was wrong. A human reviewer (or the automatic-flags check) can
    then see exactly where to check the source PDF."""
    original = value
    for bad, good in sorted(_CONTROL_CHAR_REPAIRS.items(), key=lambda kv: -len(kv[0])):
        value = value.replace(bad, good)
    if any(c in _STRIP_CONTROL_CHARS for c in value):
        value = "".join((_UNRECOVERABLE_MARKER if c in _STRIP_CONTROL_CHARS else c) for c in value)
    return value, value != original


def sanitize_json(obj: Any, path: str = "$") -> Any:
    """Recursively sanitizes string values in a parsed JSON structure,
    logging every field that had to be repaired so it stays auditable."""
    if isinstance(obj, str):
        cleaned, changed = sanitize_text(obj)
        if changed:
            logger.warning("sanitized control character(s) in LLM output at %s: %r -> %r", path, obj, cleaned)
        return cleaned
    if isinstance(obj, list):
        return [sanitize_json(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: sanitize_json(v, f"{path}.{k}") for k, v in obj.items()}
    return obj


class LLMCallError(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict
    model: str
    prompt_version: str
    created_at: str
    cached: bool
    raw_usage: Optional[dict] = None


class LLMClient:
    def __init__(self, provider: str = "openai", dry_run: bool = False, use_cache: bool = True):
        self.provider = provider
        self.dry_run = dry_run
        self.use_cache = use_cache
        self._client = None
        if not dry_run:
            self._client = self._make_client(provider)
        ensure_dir(CACHE_DIR)

    @staticmethod
    def _make_client(provider: str):
        if provider != "openai":
            raise NotImplementedError(f"provider '{provider}' is not wired yet")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMCallError(
                "openai package is required for live calls: pip install openai"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMCallError(
                "OPENAI_API_KEY is not set. Export it or add it to sb_literature_pipeline/.env, "
                "or pass dry_run=True."
            )
        return OpenAI(api_key=api_key)

    def _cache_path(self, cache_key: str) -> Path:
        return CACHE_DIR / f"{cache_key}.json"

    def call_structured(
        self,
        *,
        task: str,
        system_prompt: str,
        user_payload: dict,
        json_schema: dict,
        schema_name: str,
        model: str,
        prompt_version: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        stub_fn: Optional[Callable[[dict], dict]] = None,
    ) -> LLMResult:
        """Run one structured call. `user_payload` follows the call contract
        in plan §8.1 (task, paper_id, compound_id, aliases, instructions,
        evidence_units, output_schema)."""

        if self.dry_run:
            if stub_fn is None:
                raise LLMCallError(f"dry_run call for task '{task}' requires a stub_fn")
            return LLMResult(
                data=stub_fn(user_payload),
                model="dry-run",
                prompt_version=prompt_version,
                created_at=now_iso(),
                cached=False,
            )

        cache_key = sha256_of(
            {"task": task, "model": model, "prompt_version": prompt_version, "payload": user_payload}
        )
        if self.use_cache:
            cpath = self._cache_path(cache_key)
            if cpath.exists():
                cached = json.loads(cpath.read_text())
                logger.debug("cache hit task=%s key=%s", task, cache_key)
                return LLMResult(
                    data=cached["data"],
                    model=cached["model"],
                    prompt_version=prompt_version,
                    created_at=cached["created_at"],
                    cached=True,
                )

        user_content = json.dumps(user_payload, indent=2, default=str)
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema": json_schema,
                            "strict": True,
                        },
                    },
                )
                content = resp.choices[0].message.content
                data = sanitize_json(json.loads(content), path=f"${task}")
                created_at = now_iso()
                if self.use_cache:
                    self._cache_path(cache_key).write_text(
                        json.dumps({"data": data, "model": model, "created_at": created_at})
                    )
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    data=data,
                    model=model,
                    prompt_version=prompt_version,
                    created_at=created_at,
                    cached=False,
                    raw_usage=usage.model_dump() if usage else None,
                )
            except Exception as exc:  # noqa: BLE001 — bounded retry, then raise
                last_err = exc
                logger.warning("task=%s attempt=%d/%d failed: %s", task, attempt, max_retries, exc)
        raise LLMCallError(f"task '{task}' failed after {max_retries} attempts: {last_err}")
