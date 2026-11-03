"""
Model Context Protocol (MCP) tool surface for LLM Semantic Cache,
Multi-Tenant Token-Budget Governance, and Delta-Feedback Length Validation.
"""

from __future__ import annotations

from typing import Any, Dict

from llm_gateway_router.cache import WriteThroughSemanticCache
from llm_gateway_router.length_gate import CeilingLengthGate

_cache = WriteThroughSemanticCache(":memory:")
_gate = CeilingLengthGate()


def mcp_validate_passage_word_budget(
    task_type: str, generated_text: str
) -> Dict[str, Any]:
    """Validates generated text against task word-count ceiling/floor specs and emits delta retry instructions."""
    return _gate.validate(task_type, generated_text).model_dump()


def mcp_fetch_or_warm_translation_cache(
    namespace: str, target_locale: str, source_text: str
) -> Dict[str, Any]:
    """Fetches a SHA-256 normalized string from the write-through cache or generates with quality-gate escalation."""
    return _cache.get_or_generate(namespace, target_locale, source_text).model_dump()
