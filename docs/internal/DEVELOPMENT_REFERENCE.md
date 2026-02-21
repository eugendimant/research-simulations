# Development Reference — Historical Context & Completed Plans

This document contains historical development context, completed improvement plans,
and detailed architecture notes moved from CLAUDE.md to reduce token overhead.
Consult this when doing deep work on specific subsystems.

---

## Development Philosophy

Complex systems improve best through focused iterations, not big-bang rewrites.
Each iteration should focus on one area, build on previous work, and maintain stability.

---

## Streamlit DOM & Navigation — Hard-Won Lessons

- `st.markdown('<div>')` does NOT wrap subsequent widgets (auto-closed by browser)
- `st.container(height=N)` creates clipped container (height=1 = invisible)
- `_st_components.html()` creates an iframe; JS inside persists via MutationObserver
- NEVER use `window.parent.location.href` — destroys WebSocket + session state
- Widget execution order: widgets below set state AFTER code above reads it. Fix with `st.container()` placeholder at top, fill at bottom.
- Builder vs QSF paths have different session_state keys

### Failed navigation approaches (DO NOT retry):
1. Hidden buttons + MutationObserver (v1.0.3.4) — leaked into UI
2. Query-param navigation (v1.0.3.3) — full reload lost state
3. CSS wrapper hiding (v1.0.3.1-3.2) — Streamlit renders buttons as siblings
4. MutationObserver + setInterval hiding (v1.0.3.2) — timing issues

---

## OE Response Generation History

### v1.0.3.8-v1.0.3.10: Initial overhaul
- Fixed preview generator (was hardcoded generic), ComprehensiveResponseGenerator (was always "explanation" type), careless responses (now on-topic), consumer language defaults, condition modifiers (now domain-gated)

### v1.0.4.8: Behavioral Coherence Pipeline
- `_build_behavioral_profile()` computes 7D trait vector
- `_enforce_behavioral_coherence()` validates text-numeric consistency
- LLM prompts include BEHAVIOR hints and RULE #9 (consistency)
- Straight-liners truncated, sentiment polarity corrected

### v1.0.5.0: Deep OE Realism
- 7-strategy topic extraction (entity extraction, intent classification, study title)
- Full persona traits in all generators
- Cross-response consistency via `_participant_voice_memory`

### v1.1.0.2: 5x Non-LLM Quality (Current)
- 8 structural archetypes replacing rigid linear formula
- Domain-specific concrete detail banks (7 domains)
- Natural imperfection engine (typos, fragments, missing punctuation)
- Topic naturalization (pronoun substitution after first mention)
- Telltale phrase removal from all phrase banks

---

## Completed Improvement Plans

### v1.0.4.5: Simulation Realism (COMPLETED)
- 18 STEP 2 domains with 5+ rules
- 52+ domain personas, 13 persona × condition interactions
- Reverse-item engagement failure, SD domain sensitivity

### v1.0.4.6: Pipeline Quality Overhaul (COMPLETED)
- Domain-aware STEP 2 routing, trait modifiers, persona weights, calibration
- Effect stacking guard, persona pool validation, cross-DV coherence

---

## Remaining Improvement Targets

1. Narrative transportation domain (Green & Brock 2000)
2. Scale type detection expansion (matrix, forced choice, semantic differential)
3. LLM response validation (off-topic detection, meta-commentary screening)
4. Authority/NFC persona interactions in STEP 3

---

## Business Brainstorm

### Priority order:
1. User Accounts + Workspaces (prerequisite)
2. Tiered Pricing (Free/$29/$99/Enterprise)
3. LMS Integration (Canvas/Moodle/Blackboard via LTI)
4. REST API + SDK
5. Study Template Marketplace

### Revenue projection: ~$90K Y1, ~$440K Y2

---

## Scientific References

| Topic | Reference | Effect |
|-------|-----------|--------|
| Dictator game baseline | Engel (2011) | mean ~28% |
| Political polarization | Dimant (2024) | d = 0.6-0.9 |
| Intergroup discrimination | Iyengar & Westwood (2015) | economic games |
| Reverse item failure | Woods (2006) | 10-15% ignore |
| SD domain sensitivity | Nederhof (1985) | d = 0.25-0.75 |
| Stereotype threat | Nguyen & Ryan (2008) | d = 0.26 |
| Sunk cost | Arkes & Blumer (1985) | d = 0.30-0.50 |

## LLM Provider Chain (priority order, v1.2.1.4)
1. Google AI Gemini 2.5 Flash (15 RPM, 1M TPM — high-quality volume)
2. Google AI Gemini 2.5 Flash Lite (30 RPM, 250K TPM — cost-efficient)
3. Groq Llama 3.3 70B (~30 RPM, 14,400 RPD)
4. Cerebras Llama 3.3 70B (~30 RPM, 1M tokens/day)
5. SambaNova Llama 3.1 70B (20 RPM, persistent free tier)
6. Mistral AI Mistral Small (2 RPM, 1B tokens/month)
7. OpenRouter Mistral Small 3.1 (varies)
