# Memory

## 2026-02-16 — LLM Generation Anti-Hang Overhaul (v1.1.1.0 → v1.1.1.2)

### Critical Bug Fixed: 12-Hour Generation Hang
Free LLM generation could hang indefinitely even with available API tokens. Root causes:
1. **Auto-recovery defeated budget enforcement**: `is_llm_available` re-enabled dead providers every 20s because `_api_disabled_time` was set from a previous batch failure. Budget enforcement set `_api_available = False` but auto-recovery undid it.
2. **Quality filter silently rejected ALL valid LLM responses**: Topic keyword matching required exact substring match — "I thought it was great" rejected for "overall_experience" because it lacks "overall" keyword. Every batch appeared to "fail" even though API returned valid data.
3. **Rate limiter timestamp bug**: `wait_if_needed()` returned without appending timestamp when sleep > 15s, corrupting rate limiter state.
4. **Prefill budget too short (30s)**: Only 2/15 pool buckets filled → 500+ responses needed expensive on-demand generation (each 20s+ per provider).
5. **Batch retry wasted time**: Tried all 4 batch sizes [20,10,5,1] even when first 2 failed.

### Five-Layer Safety System Implemented
1. `_force_disabled` flag on LLMResponseGenerator — `disable_permanently()` prevents auto-recovery from re-enabling
2. Cumulative failure counter (never resets on success) — permanent disable after 15 total failures
3. Per-participant timeout tracking — 3 consecutive participants >45s → permanent disable
4. OE generation budget (180s) now uses `disable_permanently()`
5. Global watchdog daemon thread — 10-min hard timeout + 120s stall detection

### User-Facing Improvements
- **Pre-flight health check**: Tests AI providers BEFORE generation starts. Shows 3 choices if failed (retry / own API key / template)
- **Real-time progress**: Shows elapsed time, LLM stats (AI responses vs template), per-OE-question progress
- **Post-generation data source breakdown**: "AI-generated: 65% (130) | Template: 35% (70)" with options to re-generate
- Progress callback fires EVERY participant (was every 5%)

### Pipeline Efficiency Fixes
- Quality filter uses 3-char prefix matching + accepts all-rejected batches
- Rate limiter returns bool; caller skips on False
- Prefill budget: 30s → 90s
- Batch retry breaks after 2 consecutive empties
- Pool draw quality rejection → accept pool response (LLM-generated) instead of triggering on-demand
- `_oe_budget_switched_count` now properly incremented

### Key Lessons
- **NEVER set `_api_available = False` directly** — always use `disable_permanently()`
- **NEVER trust topic-keyword matching** to reject LLM responses — the LLM was prompted correctly
- **User MUST always see** what data source their text came from (AI vs template)
- **Progress must update EVERY participant** during OE — users staring at a stale spinner will assume the app is broken
- **Pre-flight checks save hours** — testing the API before generation catches 90% of issues instantly

---

## 2026-02-13 — Generation Method Chooser + SocSim Integration (v1.0.8.1)

- 4-option generation method chooser UI with expandable info tooltips:
  - Option 1: Built-in AI (free LLM providers) — Recommended
  - Option 2: User's own API key (6 providers, auto-detection, format validation)
  - Option 3: Built-in template engine (225+ domains, 58+ personas)
  - Option 4: Experimental SocSim engine (Fehr-Schmidt, IRT, 28 games)
- Real-time progress counter replacing static time estimates:
  - Phase-specific updates (personas, scales, OE, generating, socsim_enrichment)
  - Shows current/total, percentage, progress bar, elapsed time, ETA
- SocSim v0.16 integrated as experimental feature:
  - New `utils/socsim_adapter.py` bridge module
  - Game DV auto-detection (12 game types via regex patterns)
  - Condition→topic mapping (ingroup/outgroup/anonymous/etc.)
  - Engine `use_socsim_experimental` parameter wired from UI→engine
  - Post-generation enrichment: standard sim runs first, SocSim enriches game DVs
- API key visual format validation (6 providers, green/red indicators)
- Updated to v1.0.8.1.

## 2026-02-13 — Massive OE template expansion (v1.0.8.0)

- 20 iterations of comprehensive template-based open-text response improvements across 3 files:
  - `response_library.py`: Domain vocabulary expanded from ~15 to 40+ categories; `_extend()` expanded with 50+ StudyDomain entries; `_personalize_for_question()` enhanced with 8 new domain condition modifiers; action verbs 15→33, object/targets 8→24, key phrases 5→15; `_make_careless()` 8→25 templates; intent detection expanded; intensifiers/qualifiers doubled; general extensions and codas expanded
  - `persona_library.py`: All 5 question-type template banks (opinion, explanation, feeling, description, evaluation) expanded across all persona styles (engaged, satisficer, extreme, careless, default); SD hedges 3→6; follow-up thoughts 5→8 per sentiment
  - `app.py`: Preview system synced with main engine — phrase patterns 4→11, cap 60→150 chars, intent detection added (6 categories), intent-aware preview cores, elaborations reference actual subject_phrase
- Guiding principle: domain inclusivity across all 225+ research domains — no bias toward political/identity topics
- Updated synchronized app/utils versions to `1.0.8.0`.

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
