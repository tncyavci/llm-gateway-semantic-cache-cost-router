from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Callable, Optional

from .models import CacheEntry, CacheLookupResponse, ModelTier

_WS_RE = re.compile(r"\s+")


def normalize_source_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def compute_sha256_cache_key(
    namespace: str, target_locale: str, text: str
) -> str:
    norm = normalize_source_text(text)
    payload = f"{namespace.strip().lower()}|{target_locale.strip().lower()}|{norm}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WriteThroughSemanticCache:
    """
    SQLite-backed SHA-256 deterministic content & localization cache with
    two-tier quality-gate model escalation:
      - First checks the normalized SHA-256 cache (`namespace|locale|normalized_text`).
      - On a cache miss, invokes the low-cost `MINI_FAST` generator first.
      - Runs a deterministic quality check (e.g., suspect untranslated markers,
        placeholder leakage, or abnormal length ratio). If flagged as suspect,
        automatically escalates to `FLAGSHIP` before persisting to the cache.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS content_cache (
                sha256_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                target_locale TEXT NOT NULL,
                source_text_normalized TEXT NOT NULL,
                output_text TEXT NOT NULL,
                model_used TEXT NOT NULL,
                escalated INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def get_or_generate(
        self,
        namespace: str,
        target_locale: str,
        source_text: str,
        mini_generator: Optional[Callable[[str, str], str]] = None,
        flagship_generator: Optional[Callable[[str, str], str]] = None,
    ) -> CacheLookupResponse:
        norm = normalize_source_text(source_text)
        key = compute_sha256_cache_key(namespace, target_locale, norm)

        row = self._conn.execute(
            """
            SELECT sha256_key, namespace, target_locale, source_text_normalized,
                   output_text, model_used, escalated
              FROM content_cache
             WHERE sha256_key = ?
            """,
            (key,),
        ).fetchone()

        if row:
            return CacheLookupResponse(
                hit=True,
                entry=CacheEntry(
                    sha256_key=row[0],
                    namespace=row[1],
                    target_locale=row[2],
                    source_text_normalized=row[3],
                    output_text=row[4],
                    model_used=ModelTier(row[5]),
                    escalated_by_quality_gate=bool(row[6]),
                ),
            )

        gen_mini = mini_generator or (
            lambda txt, loc: f"[{loc}:{ModelTier.MINI_FAST.value}] {txt}"
        )
        gen_flagship = flagship_generator or (
            lambda txt, loc: f"[{loc}:{ModelTier.FLAGSHIP.value}] {txt}"
        )

        candidate_out = gen_mini(norm, target_locale)
        escalated = False
        model_used = ModelTier.MINI_FAST

        if self._is_suspect_output(norm, candidate_out):
            candidate_out = gen_flagship(norm, target_locale)
            model_used = ModelTier.FLAGSHIP
            escalated = True

        entry = CacheEntry(
            sha256_key=key,
            namespace=namespace,
            target_locale=target_locale,
            source_text_normalized=norm,
            output_text=candidate_out,
            model_used=model_used,
            escalated_by_quality_gate=escalated,
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO content_cache
            (sha256_key, namespace, target_locale, source_text_normalized, output_text, model_used, escalated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.sha256_key,
                entry.namespace,
                entry.target_locale,
                entry.source_text_normalized,
                entry.output_text,
                entry.model_used.value,
                1 if entry.escalated_by_quality_gate else 0,
            ),
        )
        self._conn.commit()
        return CacheLookupResponse(hit=False, entry=entry)

    @staticmethod
    def _is_suspect_output(source_norm: str, output: str) -> bool:
        out_clean = output.strip()
        if not out_clean:
            return True
        if "TODO" in out_clean or "[UNTRANSLATED]" in out_clean or "???" in out_clean:
            return True
        # Extreme truncation heuristic
        if len(source_norm) >= 30 and len(out_clean) < (len(source_norm) * 0.25):
            return True
        return False
