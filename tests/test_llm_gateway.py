from __future__ import annotations

from fastapi.testclient import TestClient

from app import app
from llm_gateway_router.budget import TokenBudgetGovernor
from llm_gateway_router.cache import (
    WriteThroughSemanticCache,
    compute_sha256_cache_key,
)
from llm_gateway_router.length_gate import CeilingLengthGate
from llm_gateway_router.models import ModelTier


def test_sha256_cache_normalization_and_write_through_hit() -> None:
    k1 = compute_sha256_cache_key("lessons", "fa", "  Hello   world \n ")
    k2 = compute_sha256_cache_key("LESSONS", "FA", "Hello world")
    assert k1 == k2

    cache = WriteThroughSemanticCache(":memory:")
    r1 = cache.get_or_generate("lessons", "fa", "Patient needs immediate oxygen monitoring.")
    assert r1.hit is False
    assert r1.entry.model_used == ModelTier.MINI_FAST
    assert r1.entry.escalated_by_quality_gate is False

    # Second call with extra whitespace hits the cache without invoking generator
    r2 = cache.get_or_generate("lessons", "fa", "  Patient   needs immediate oxygen monitoring. ")
    assert r2.hit is True
    assert r2.entry.sha256_key == r1.entry.sha256_key


def test_quality_gate_escalates_suspect_mini_output_to_flagship() -> None:
    cache = WriteThroughSemanticCache(":memory:")
    res = cache.get_or_generate(
        namespace="lessons",
        target_locale="de",
        source_text="Administer anticoagulant therapy according to protocol.",
        mini_generator=lambda txt, loc: "TODO [UNTRANSLATED]",
        flagship_generator=lambda txt, loc: "Antikoagulationstherapie gemäß Protokoll verabreichen.",
    )
    assert res.hit is False
    assert res.entry.escalated_by_quality_gate is True
    assert res.entry.model_used == ModelTier.FLAGSHIP
    assert "Antikoagulationstherapie" in res.entry.output_text


def test_ceiling_length_gate_interpolates_ceiling_and_computes_exact_delta() -> None:
    gate = CeilingLengthGate()
    prompt = gate.build_generation_prompt("long_form_passage", "Cloud Incident Response")
    # Prompt must name the ceiling (500 words) to counteract 0.8x LLM undershoot
    assert "Target length: 500 words" in prompt

    short_text = " ".join(["word"] * 280)
    val_short = gate.validate("long_form_passage", short_text)
    assert val_short.in_range is False
    assert val_short.word_count == 280
    assert val_short.word_delta_to_midpoint == 170  # midpoint is 450
    assert val_short.retry_feedback_instruction is not None
    assert "+170 words" in val_short.retry_feedback_instruction

    valid_text = " ".join(["word"] * 435)
    val_ok = gate.validate("long_form_passage", valid_text)
    assert val_ok.in_range is True
    assert val_ok.retry_feedback_instruction is None


def test_token_budget_governor_weights_models_and_enforces_429() -> None:
    gov = TokenBudgetGovernor(":memory:", default_weekly_cap=1000)
    # 100 flagship tokens * 8.0 weight = 800 weighted tokens
    b1 = gov.record_usage("tenant-a", 100, ModelTier.FLAGSHIP, "2026-W45")
    assert b1.used_weighted_tokens == 800
    assert b1.within_limit is True
    assert b1.remaining_weighted_tokens == 200

    # 30 vision tokens * 10.0 weight = 300 weighted tokens -> total 1100 >= 1000 cap
    b2 = gov.record_usage("tenant-a", 30, ModelTier.VISION_HIGH_RES, "2026-W45")
    assert b2.used_weighted_tokens == 1100
    assert b2.within_limit is False
    assert b2.remaining_weighted_tokens == 0


def test_fastapi_gateway_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    c_resp = client.post(
        "/v1/cache/translate-or-fetch",
        json={
            "tenant_id": "t-1",
            "namespace": "ui",
            "target_locale": "de",
            "source_text": "Verify zero energy state before maintenance.",
        },
    )
    assert c_resp.status_code == 200
    assert c_resp.json()["hit"] is False

    c_resp_hit = client.post(
        "/v1/cache/translate-or-fetch",
        json={
            "tenant_id": "t-1",
            "namespace": "ui",
            "target_locale": "de",
            "source_text": "Verify   zero energy state before maintenance.",
        },
    )
    assert c_resp_hit.status_code == 200
    assert c_resp_hit.json()["hit"] is True

    l_resp = client.post(
        "/v1/length-gate/validate",
        json={
            "task_type": "executive_brief",
            "generated_text": " ".join(["summary"] * 200),
        },
    )
    assert l_resp.status_code == 200
    assert l_resp.json()["in_range"] is True
