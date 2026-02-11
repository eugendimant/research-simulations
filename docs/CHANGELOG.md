# Agent Change Log

## 2026-02-11 — v1.0.4.4
### Behavioral Simulation Realism (3 Deep Iterations)

#### Iteration 1: Reverse-Item Modeling + Study Context + Domain Expansion
- **Enhanced reverse-coded item modeling (STEP 5)**: Careless/satisficing respondents now probabilistically fail to reverse items based on attention level (Woods 2006: 10-15% ignore directionality). Previously, all respondents correctly reversed items — now engagement-dependent reversal accuracy creates realistic inconsistency patterns.
- **Study context integration in trait modifiers**: Study title/description now detected for domain-level priming. Political studies prime extremity (+0.06) even in control conditions. Health studies prime social desirability (+0.04). Moral studies prime evaluative extremity (+0.05). Based on Bargh et al. (1996): Category priming affects behavior automatically.
- **STEP 2 domain expansion**: Added 30+ new keyword→effect mappings:
  - Domain 15 (Embodiment): Warmth/coldness priming (Williams & Bargh 2008), arm flexion/extension (Casasanto & Dijkstra 2010), physical touch (Crusco & Wetzel 1984), head movement (Wells & Petty 1980)
  - Domain 16 (Time): Future time perspective (Zimbardo & Boyd 1999), temporal landmarks / fresh start effect (Dai et al. 2014), nostalgia induction (Wildschut et al. 2006)
  - Domain 17 (Deception): Honor code effects (Mazar et al. 2008), monitoring/observability (Bateson et al. 2006), moral licensing (Merritt et al. 2010), self-serving justification (Shalvi et al. 2011)
  - Domain 18 (Power/Status): Power priming (Galinsky et al. 2003), social status (Kraus et al. 2012), accountability (Lerner & Tetlock 1999)

#### Iteration 2: Persona Library Expansion + Domain-Condition Interactions
- **8 new domain-specific personas** (total: 50+):
  - Anxious Individual (Clark & Watson 1991 tripartite model)
  - Resilient Individual (Connor & Davidson 2003 CD-RISC)
  - Justice-Oriented Individual (Tyler 2006 procedural justice)
  - Competitive Achiever (Vealey 1986 sport confidence)
  - Anxiously Attached (Brennan et al. 1998 ECR anxiety)
  - Avoidantly Attached (Brennan et al. 1998 ECR avoidance)
  - Overconfident Decision Maker (Barber & Odean 2001)
  - Media-Literate Individual (Friestad & Wright 1994 PKM)
- **7 new domain-condition interaction patterns in STEP 3**:
  - Power/hierarchy: Increases extremity, reduces acquiescence (Keltner 2003)
  - Competition: Increases extremity + engagement (Deutsch 1949)
  - Mindfulness/reflection: Increases attention, reduces extremity (Brown & Ryan 2003)
  - Accountability: Increases accuracy motivation (Lerner & Tetlock 1999)
  - Goal-setting: Increases engagement + consistency (Locke & Latham 2002)
  - Depletion/fatigue: Reduces attention + consistency (Baumeister 1998)
  - Mortality salience: Increases extremity + engagement (Greenberg 1990 TMT)

#### Iteration 3: Domain-Sensitive Social Desirability + Personalization
- **Domain-sensitive social desirability (STEP 9)**: SD bias now varies by construct sensitivity. Sensitive topics (prejudice, aggression, substance use) get 1.5× SD effect. Vulnerability topics (anxiety, depression, loneliness) get inverted SD (-0.5×, suppresses honest negatives). Factual/behavioral reports get 0.5× (hard to fake). Based on Nederhof (1985 meta): SD bias d = 0.25-0.75 for sensitive topics.
- **Expanded personalization modifiers**: `_personalize_for_question()` now handles 8 domain-specific condition contexts: political (progressive/conservative lean), health (risk/prevention), moral/ethical, intergroup (ingroup/outgroup), financial (gain/loss), environmental (sustainable/harmful). Previously only AI, hedonic, and utilitarian.
- **CLAUDE.md updated** with detailed improvement plan, audit findings, and implementation roadmap

### Version Synchronization
- v1.0.4.4 across: app.py, utils/__init__.py, qsf_preview.py, response_library.py, README.md, enhanced_simulation_engine.py

## 2026-02-10 — v1.0.1.4
### Bug Fixes
- Fixed open-ended question removal bug: removing a question no longer closes the expander or scrolls to top (replaced `_navigate_to(2)` with `st.rerun()`)
- Fixed "Continue to Generate" button not appearing on Design page: added inline Continue button within design validation so users can proceed immediately without needing another interaction
- Builder path OE removal also fixed to avoid expander collapse

### New Features
- **Remove All** button for open-ended questions (both QSF-detected and survey builder paths) — removes all OE questions at once when dealing with large numbers
- Inline "Continue to Generate" button appears right after design validation passes, eliminating the timing issue where the top nav button wasn't visible

### Documentation & Branding
- Removed "University of Pennsylvania" references from footer, README, and technical methods
- Updated technical methods document to reflect current LLM provider chain (Google AI Studio → Groq → Cerebras → OpenRouter)
- Regenerated methods_summary.pdf with updated content and version

## 2026-02-10 — v1.9.1
### Bug Fixes
- Fixed analytics dashboard NameError on `clean_scales` — recovered from session state
- Fixed question removal (X button) — session state flag approach replaces unreliable in-render-loop removal
- Fixed live data preview — now shows all columns/scales (removed 5-scale and 2-OE limits)
- Fixed title truncation — full title display via HTML rendering (was truncated at 30 chars)
- Fixed treatment logic — unmatched blocks in BlockRandomizer no longer default to visible for all conditions

### LLM Pipeline (Critical)
- Improved error diagnostics: HTTP status codes, auth errors (401/403), rate limits (429), connection errors logged explicitly
- Added 5-provider failover chain: Groq → Cerebras → Together AI → SambaNova → OpenRouter
- Added 8 fallback response parsing strategies (truncated JSON recovery, numbered lists, newline-delimited)
- Added retry with exponential backoff (up to 3 retries per call per provider)
- Added cooldown-based auto-recovery for temporarily disabled providers
- Engine prefill gate no longer requires `is_llm_available` — always attempts LLM generation
- Providers reset after failed prefill so on-demand generation gets a fresh start

### Response Realism (4 Deep Iterations)
- Within-person coherence (g-factor) with differential construct-type loadings (Podsakoff et al., 2003)
- Personality × Condition interaction effects (ELM, Petty & Cacioppo, 1986; satisficing, Krosnick, 1991)
- Response time simulation with log-normal distributions (Callegaro et al., 2015; Yan & Tourangeau, 2008)
- Correlated response styles (acquiescence-extremity-social desirability) via Cholesky decomposition (Baumgartner & Steenkamp, 2001)

### UX Improvements
- Redesigned landing page: replaced 4 expanders with professional tabbed design
- Analytics dashboard: added prominent dark gradient header with icon
- Report: updated provider display names for Together AI and SambaNova

### Research Domain Detection
- Added 50+ missing keyword sets across research domains
- Expanded matching with better fallback logic

## 2025-01-27
- Updated landing page branding, attribution, and method summary download link.
- Added Qualtrics survey PDF upload to improve domain inference and stored excerpts in metadata.
- Created a new methods summary document outlining persona modeling and recommended uploads.
