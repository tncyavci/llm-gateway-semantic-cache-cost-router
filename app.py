from __future__ import annotations

from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_gateway_router.budget import TokenBudgetGovernor
from llm_gateway_router.cache import WriteThroughSemanticCache
from llm_gateway_router.length_gate import CeilingLengthGate
from llm_gateway_router.models import (
    BudgetCheckResult,
    CacheLookupResponse,
    LengthValidationResult,
    ModelTier,
)

app = FastAPI(
    title="LLM Gateway Semantic Cache, Budget Governor & Length Gate",
    version="1.0.0",
)

cache = WriteThroughSemanticCache(":memory:")
governor = TokenBudgetGovernor(":memory:", default_weekly_cap=1_000_000)
length_gate = CeilingLengthGate()


class CacheRequest(BaseModel):
    tenant_id: str = "default-tenant"
    iso_week: str = "2026-W45"
    namespace: str = "clinical_lessons"
    target_locale: str = "de"
    source_text: str = Field(..., min_length=1)


class UsageRecordRequest(BaseModel):
    tenant_id: str
    iso_week: str = "2026-W45"
    raw_tokens: int = Field(..., ge=0)
    model_tier: ModelTier = ModelTier.MINI_FAST


class LengthCheckRequest(BaseModel):
    task_type: str = "long_form_passage"
    generated_text: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/cache/translate-or-fetch", response_model=CacheLookupResponse)
def translate_or_fetch(req: CacheRequest) -> CacheLookupResponse:
    budget = governor.check_budget(req.tenant_id, req.iso_week)
    if not budget.within_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Weekly token budget exceeded for tenant '{req.tenant_id}'.",
        )
    res = cache.get_or_generate(
        namespace=req.namespace,
        target_locale=req.target_locale,
        source_text=req.source_text,
    )
    if not res.hit:
        governor.record_usage(
            tenant_id=req.tenant_id,
            raw_tokens=max(10, len(req.source_text.split()) * 2),
            model_tier=res.entry.model_used,
            iso_week=req.iso_week,
        )
    return res


@app.post("/v1/budget/record", response_model=BudgetCheckResult)
def record_budget_usage(req: UsageRecordRequest) -> BudgetCheckResult:
    return governor.record_usage(
        tenant_id=req.tenant_id,
        raw_tokens=req.raw_tokens,
        model_tier=req.model_tier,
        iso_week=req.iso_week,
    )


@app.post("/v1/length-gate/validate", response_model=LengthValidationResult)
def validate_length(req: LengthCheckRequest) -> LengthValidationResult:
    if req.task_type not in length_gate.specs:
        raise HTTPException(status_code=404, detail="Unknown task_type")
    return length_gate.validate(req.task_type, req.generated_text)
