# Memory

## 2026-02-12 — LLM-first simulation reliability overhaul (v1.0.6.8)

- Enforced explicit fallback consent behavior for open-ended generation:
  - If LLM providers are unavailable, users must either provide a key or explicitly allow one-run final fallback.
  - Without either, generation is blocked in LLM-first mode.
- Added runtime plumbing so `allow_template_fallback` flows from UI → engine → `LLMResponseGenerator`.
- Hardened `LLMResponseGenerator` to raise an error when all providers fail and fallback is disabled.
- Fixed critical condition corruption bug:
  - OE duplicate post-processing previously ran on all string-list columns and mutated `CONDITION` / `_PERSONA`.
  - Dedup now runs only on known open-ended columns.
- Added instructor report condition canonicalization to map contaminated labels (e.g., prefixed text) back to metadata conditions when unambiguous.
- Updated synchronized app/utils versions to `1.0.6.8`.


## 2026-02-12 — v1.0.6.9 follow-up
- Fixed LLM init regression (`os` import) that caused admin to report template-only runs.
- Reordered provider chain to put Google AI Studio first.
- Added OE quality filters + topical relevance checks to reduce gibberish.
- Added DV micro-dynamics (fatigue drift, streak inertia, endpoint correction) for more human-like item patterns.
- Added detailed roadmap doc: `docs/llm_and_behavior_overhaul_plan.md`.
