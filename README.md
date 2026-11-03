# Deterministic LLM Cost-Router, SHA-256 Write-Through Cache & Self-Healing Length Gate

[![CI](https://github.com/tncyavci/llm-gateway-semantic-cache-cost-router/actions/workflows/ci.yml/badge.svg)](https://github.com/tncyavci/llm-gateway-semantic-cache-cost-router/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A production-grade LLM Gateway & Content Generation Guardrail Middleware implementing SHA-256 Write-Through Localization/Content Caching with Quality-Gate Model Escalation (`gpt-4o-mini` $\to$ `gpt-4o`), ISO-Week Multi-Tenant Weighted Token-Budget Governance, and Ceiling-Targeted Delta-Feedback Length Validation.**

Built to solve three high-cost operational bottlenecks in multi-lingual and content-heavy Applied AI SaaS platforms:
1. **Live Translation & Redundant Generation Drain:** Calling an LLM at request time for repeated UI/curriculum strings adds 800ms+ latency and recurring token cost. A whitespace-normalized SHA-256 write-through cache with automatic quality-gate escalation (`mini` $\to$ `flagship` on suspect outputs) serves 99%+ of reads from SQLite/Postgres in $<1\text{ms}$.
2. **Systematic LLM Word-Count Undershoot ($\approx 0.8\times$):** When prompted for a range such as "400–500 words", frontier models systematically anchor near $0.8\times$ of the lower bound ($\approx 280\text{–}320$ words). Naming the **ceiling** (`500 words`) in the prompt and retrying misses with the exact **concrete word delta** (`+135 words`) rather than hotter temperature brings pass rates to ~100%.
3. **Unbounded Multi-Modal Token Spend:** Audio transcription, high-res vision, and flagship reasoning models have vastly different unit costs. The `TokenBudgetGovernor` enforces ISO-week tenant caps using model-tier cost weights with fail-open telemetry persistence and HTTP 429 enforcement.

---

## System Architecture

```mermaid
flowchart LR
    A["Client / Pipeline Request"] --> B{"TokenBudgetGovernor\n(ISO-Week Weighted Cap)"}
    B -- "Over Cap" --> C["HTTP 429 Rate Limit"]
    B -- "Within Cap" --> D{"WriteThroughSemanticCache\n(SHA-256 Normalized Key)"}
    D -- "Cache Hit" --> E["Return Cached Output\n(0 LLM Tokens)"]
    D -- "Cache Miss" --> F["Invoke Tier-1 (gpt-4o-mini)"]
    F --> G{"Quality & Suspect Gate\n(Truncation / Placeholder Check)"}
    G -- "Suspect" --> H["Escalate to Tier-2 (gpt-4o)"]
    G -- "Clean" --> I["Persist to SQLite Cache"]
    H --> I
    I --> J["CeilingLengthGate\n(Word Count & Concrete Delta Feedback)"]
```

---

## Quickstart & Testing

```bash
pip install -e ".[dev]"
pytest -v
uvicorn app:app --reload --port 8000
```
