# Memory

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
