from __future__ import annotations

import sqlite3
from math import ceil

from .models import MODEL_TOKEN_WEIGHTS, BudgetCheckResult, ModelTier


class TokenBudgetGovernor:
    """
    ISO-week keyed multi-tenant LLM token usage governor with model-tier cost
    weighting and fail-open telemetry resilience.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        default_weekly_cap: int = 1_000_000,
    ) -> None:
        self.default_weekly_cap = default_weekly_cap
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_usage_ledger (
                tenant_id TEXT NOT NULL,
                iso_week TEXT NOT NULL,
                used_weighted_tokens INTEGER NOT NULL DEFAULT 0,
                weekly_cap_tokens INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, iso_week)
            )
            """
        )
        self._conn.commit()

    def check_budget(
        self, tenant_id: str, iso_week: str = "2026-W45"
    ) -> BudgetCheckResult:
        row = self._conn.execute(
            """
            SELECT used_weighted_tokens, weekly_cap_tokens
              FROM ai_usage_ledger
             WHERE tenant_id = ? AND iso_week = ?
            """,
            (tenant_id, iso_week),
        ).fetchone()

        used = row[0] if row else 0
        cap = row[1] if row else self.default_weekly_cap
        remaining = max(0, cap - used)
        return BudgetCheckResult(
            tenant_id=tenant_id,
            iso_week=iso_week,
            within_limit=(used < cap),
            used_weighted_tokens=used,
            weekly_cap_tokens=cap,
            remaining_weighted_tokens=remaining,
        )

    def record_usage(
        self,
        tenant_id: str,
        raw_tokens: int,
        model_tier: ModelTier,
        iso_week: str = "2026-W45",
    ) -> BudgetCheckResult:
        weight = MODEL_TOKEN_WEIGHTS.get(model_tier, 1.0)
        weighted_delta = max(0, ceil(raw_tokens * weight))

        try:
            self._conn.execute(
                """
                INSERT INTO ai_usage_ledger (tenant_id, iso_week, used_weighted_tokens, weekly_cap_tokens)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id, iso_week)
                DO UPDATE SET used_weighted_tokens = used_weighted_tokens + excluded.used_weighted_tokens
                """,
                (tenant_id, iso_week, weighted_delta, self.default_weekly_cap),
            )
            self._conn.commit()
        except sqlite3.Error:
            # Fail-open metering: never crash caller on telemetry storage blip
            pass

        return self.check_budget(tenant_id, iso_week)
