from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    MINI_FAST = "gpt-4o-mini"
    FLAGSHIP = "gpt-4o"
    AUDIO_TRANSCRIBE = "gpt-4o-transcribe"
    VISION_HIGH_RES = "gpt-4o-vision"


MODEL_TOKEN_WEIGHTS: Dict[ModelTier, float] = {
    ModelTier.MINI_FAST: 1.0,
    ModelTier.AUDIO_TRANSCRIBE: 4.0,
    ModelTier.FLAGSHIP: 8.0,
    ModelTier.VISION_HIGH_RES: 10.0,
}


class CacheEntry(BaseModel):
    sha256_key: str
    namespace: str
    target_locale: str
    source_text_normalized: str
    output_text: str
    model_used: ModelTier
    escalated_by_quality_gate: bool = False


class CacheLookupResponse(BaseModel):
    hit: bool
    entry: CacheEntry


class BudgetCheckResult(BaseModel):
    tenant_id: str
    iso_week: str
    within_limit: bool
    used_weighted_tokens: int
    weekly_cap_tokens: int
    remaining_weighted_tokens: int


class LengthGateSpec(BaseModel):
    task_type: str
    min_words: int = Field(..., gt=0)
    max_words: int = Field(..., gt=0)


class LengthValidationResult(BaseModel):
    task_type: str
    word_count: int
    min_words: int
    max_words: int
    prompted_ceiling_target: int
    in_range: bool
    word_delta_to_midpoint: int
    retry_feedback_instruction: Optional[str] = None
