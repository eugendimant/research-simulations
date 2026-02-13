
## 2026-02-12 — v1.0.7.1
### LLM-First Integrity + Quality Release
- Removed permissive pre-run fallback UX path and kept template fallback disabled by default.
- Added failure-time explicit one-time emergency fallback consent button after API-only failure.
- Added post-run LLM integrity guard: OE runs in strict mode must show API calls/attempts.
- Added LLM init error propagation (`llm_init_error`) from engine to metadata and report layers.
- Updated report logic (Markdown + HTML) to show strict-mode/init-failure integrity warnings instead of ambiguous template messaging.
- Expanded OE quality filtering and diagnostics for relevance/gibberish control.
- Added detailed roadmap doc: `docs/software_10x_improvements.md`.

### Human-Like Simulation Improvements
- Added within-scale micro-dynamics: fatigue drift, low-consistency inertia, and occasional endpoint correction.
- Preserved bounds and condition effect primacy while improving natural response trajectories.

### Version Synchronization
- Updated synchronized app/utils versions to v1.0.7.1.
- Added OE completeness guarantee: visible open-ended questions are no longer blanked by optional-skip logic.
- Added adaptive LLM batch fallback (20→10→5→1) to reduce failure sensitivity while preserving throughput.
- Added Admin Dashboard provider quota/cost calculator and explicit note that batch size fallback protects generation completeness.
- Improved final emergency fallback text to be contextual (question/condition-aware) instead of generic constant output.
- Updated README provider-chain description to match current Google-first multi-provider order.


## 2026-02-12 — v1.0.6.9
### Critical LLM Reliability Fixes (3 iterations)
- **Iteration 1:** Fixed LLM initialization regression in `EnhancedSimulationEngine` (`name 'os' is not defined`) by restoring `import os`.
- **Iteration 2:** Reordered built-in provider chain to prioritize **Google AI Studio first** (Gemma, then Gemini), then Groq/Cerebras/Poe/OpenRouter.
- **Iteration 3:** Added stronger diagnostics counters (`quality_rejections`) in LLM stats to improve admin visibility into response filtering.

### Open-Ended Quality Improvements (baseline + 5 refinements)
- Added topical token extraction from question+condition context.
- Added low-quality/gibberish filtering for LLM batch outputs (generic stock phrases, low diversity, off-topic text).
- Applied quality checks to pool-drawn responses as well, reducing stale generic text reuse.
- Added explicit rejection tracking to make quality filtering observable in diagnostics.

### Human-Like DV Behavior Improvements (5 implementation iterations)
- Added item-position-aware fatigue drift toward midpoint for low-attention respondents on longer scales.
- Added low-consistency streak inertia using participant-by-variable memory.
- Added occasional endpoint self-correction for high-attention/non-extreme respondents.
- Added per-item state plumbing (`_current_item_position`, `_current_item_total`) to support realistic within-scale dynamics.
- Preserved strict bounds and treatment effects while adding micro-dynamics.

### Documentation
- Added `docs/llm_and_behavior_overhaul_plan.md` with:
  - 3-iteration LLM fix plan,
  - baseline+5 OE quality plan,
  - detailed 20-step DV human-likeness roadmap,
  - note on constrained Westwood (2025 PNAS) retrieval in this runtime.

### Version Synchronization
- Updated to v1.0.6.9 across synchronized app/utils files.


## 2026-02-12 — v1.0.6.8
### Critical Fixes
- Enforced LLM-first open-ended generation gate in the Generate UI: when LLM providers are unavailable, generation is blocked unless the user either provides a key or explicitly allows one-run final fallback.
- Added `allow_template_fallback` policy plumbing from app → simulation engine → LLM generator.
- `LLMResponseGenerator` now raises a hard failure when all providers fail and fallback is disabled, preventing silent template substitution.
- Fixed a major data-integrity bug where OE dedup logic accidentally mutated `CONDITION` and `_PERSONA` columns; dedup now scopes strictly to open-ended columns.
- Added instructor report condition canonicalization to recover canonical condition labels from contaminated values when possible.

### Version Synchronization
- Updated to v1.0.6.8 across: app.py, utils/__init__.py, qsf_preview.py, response_library.py, README.md, enhanced_simulation_engine.py, llm_response_generator.py.

# Agent Change Log

## 2026-02-12 — v1.0.5.9
### Poe (poe.com) Added as Built-in LLM Provider

- **New built-in provider**: Poe (poe.com) added as 6th provider in the LLM failover chain
  - OpenAI-compatible API at `https://api.poe.com/v1/chat/completions`
  - Uses GPT-4o-mini model (~15 compute points/message)
  - Free tier: 3,000 compute points/day (~200 messages)
  - Positioned after OpenRouter in the chain (last resort before asking user for own key)
- **Built-in API key**: XOR-encoded Poe key ships with the tool for zero-config usage
- **Auto-detection**: `detect_provider_from_key()` updated to recognize Poe keys
- **User-selectable**: Poe added to provider dropdown in the API key fallback dialog
- **Env var support**: `POE_API_KEY` environment variable recognized for custom Poe keys
- **UI updates**: Provider count updated from 5 to 6 in exhaustion messages; Poe added to free provider sign-up table
- **Provider chain**: Groq → Cerebras → Google AI (Gemini) → Google AI (Gemma) → OpenRouter → Poe → (user key) → (template fallback)

---

## 2026-02-12 — v1.0.5.0
### OE Behavioral Realism Pipeline — 5 Iterations: Topic Intelligence, Persona Integration, Behavioral Coherence, Cross-Response Consistency

#### Iteration 1: Deep Topic Intelligence & Semantic Context Pipeline
- **7-strategy topic extraction** (up from 4): entity extraction, question intent classification, study title topic extraction
- Expanded phrase capture from 60 to 150 characters with 8 new extraction patterns
- **30 paradigm domain hints** (up from 9): moral, stereotype, prejudice, discrimination, persuasion, negotiation, risk, prosocial, cooperation, punishment, identity, emotion, mindfulness, stress, wellbeing, environmental, education, leadership, conformity, deception, attachment
- **Named entity extraction**: 45+ entities (people, brands, concepts) detected and passed through all generators
- **Question intent classification**: 7 types (opinion, explanation, description, emotional_reaction, evaluation, prediction, recommendation) shape response framing

#### Iteration 2: Full Persona Trait Integration into All Generators
- LLM prompt VOICE cues from full 7-dimensional trait profile (social desirability, extremity, consistency, intensity)
- New LLM MANDATORY RULE #10: Persona Voice & Trait Expression — traits must pervade entire response
- ComprehensiveResponseGenerator question-type routing: evaluation → explanation fallback chain (no longer hardcoded "explanation")
- Intent-specific context-grounded templates: 18+ new templates across 3 intents × 3 sentiments
- TextResponseGenerator enhanced style selection using full trait vector

#### Iteration 3: Deep Behavioral Coherence Enforcement
- `_enforce_behavioral_coherence()` expanded from 3 to 6 checks
- Expanded sentiment indicators: 30+ positive/negative words (up from 20)
- Intensity-driven intensifier injection with scaled probability (25%-65% based on intensity, up from flat 35%)
- Social desirability modulation: high-SD personas add qualifying hedges (Paulhus, 2002)
- Extremity-driven absolute language: replaces hedging words with absolutes for extreme responders (Greenleaf, 1992)

#### Iteration 4: Fallback Generator Full Behavioral Alignment
- TextResponseGenerator uses full behavioral context: intensity, consistency, SD, extremity, entities, question_intent, condition_framing
- Entity injection: named entities woven into responses at 60% probability
- LLM generator fallback path now passes behavioral_profile through to ComprehensiveResponseGenerator
- Domain-aware condition modifiers expanded: political (progressive/conservative), ingroup/outgroup, health (risk/benefit)

#### Iteration 5: Cross-Response Consistency & Participant Voice Memory
- `_participant_voice_memory` dict tracks established tone, prior responses, and style across OE questions
- Voice consistency hints in LLM CONTINUITY cues and fallback context dict
- Tone detection from generated text: positive/negative word counting establishes voice for subsequent questions
- Ensures same participant sounds the same across all their OE answers

---

## 2026-02-11 — v1.0.4.9
### 5 Deep Improvement Iterations: New Paradigms, Personas, SD Refinement, Validation, Documentation

#### Iteration 1: Simulation Engine — 5 New Research Paradigm Domains (STEP 2)
- **Domain 19 — Narrative Transportation**: Green & Brock (2000), van Laer et al. (2014 meta r=0.35), Appel & Richter (2007). Keywords: narrative, story, transported, immersed, absorbed, expository, first-person/third-person perspective. Also detects fictional vs real narratives, narrative perspective type.
- **Domain 20 — Social Comparison**: Festinger (1954), Gerber et al. (2018 meta d=0.20-0.50), Wheeler & Miyake (1992), Wills (1981). Keywords: upward/downward comparison, social media feed, instagram, assimilation/contrast, curated profile. Also detects social media comparison (Vogel et al. 2014).
- **Domain 21 — Gratitude & Positive Interventions**: Emmons & McCullough (2003), Davis et al. (2016 meta d=0.31), Sin & Lyubomirsky (2009 meta d=0.29), Seligman et al. (2005). Keywords: gratitude, thankful, count blessings, acts of kindness, savoring, best possible self.
- **Domain 22 — Moral Cleansing & Compensation**: Zhong & Liljenquist (2006), Sachdeva et al. (2009), Jordan et al. (2011), Tetlock et al. (2000). Keywords: moral threat, transgression, sacred value, taboo tradeoff, moral identity, commodify.
- **Domain 23 — Attention Economy & Digital Distraction**: Ward et al. (2017 JACR), Stothart et al. (2015), Ophir et al. (2009), Uncapher & Wagner (2018). Keywords: phone present, notification, multitask, digital detox, screen break.
- Updated `_DOMAIN_RELEVANCE` mapping for all 5 new domains
- New domain-aware STEP 4 scaling: positive_psychology (1.0×), narrative_persuasion (1.15×), digital_wellbeing (1.1×), moral_psychology (1.2×)

#### Iteration 2: Response Library — 5 New Domain Template Sets
- **narrative_transportation**: explanation + evaluation question types, positive/neutral/negative/very_positive/very_negative sentiments
- **social_comparison**: explanation + evaluation question types across all sentiment levels
- **gratitude_intervention**: explanation + evaluation question types with positive psychology language
- **moral_cleansing**: explanation + evaluation question types for moral self-regulation topics
- **digital_wellbeing**: explanation + evaluation question types for attention economy research

#### Iteration 3: Persona Library — 6 New Domain-Specific Personas (58+ total)
- **Narrative Thinker**: High engagement (0.85), elaboration (0.80), attention (0.88). Green & Brock (2000) Need for Narrative scale. Weight: 0.08.
- **Social Comparer**: High SD (0.72), moderate-high extremity (0.45), lower consistency (0.55). Gibbons & Buunk (1999). Weight: 0.10.
- **Grateful Optimist**: High response tendency (0.72), acquiescence (0.62), moderate extremity (0.28). McCullough et al. (2002). Weight: 0.08.
- **Moral Absolutist**: Very high extremity (0.78), engagement (0.85), consistency (0.80). Tetlock et al. (2000), Haidt (2001). Weight: 0.07.
- **Digital Native**: High reading speed (0.85), moderate attention (0.72), lower consistency (0.58). Prensky (2001), Ophir et al. (2009). Weight: 0.10.
- **Financial Deliberator**: High attention (0.88), consistency (0.78), low SD (0.38). Kahneman & Tversky (1979), Barber & Odean (2001). Weight: 0.08.

#### Iteration 4: Social Desirability Refinement + Reverse-Item Tracking + Validation
- **5 new SD construct sensitivity categories** in STEP 9:
  - Moral/sacred value self-reports: 1.4× multiplier (Tetlock 2000)
  - Gratitude/positive psychology: 1.2× (McCullough 2002)
  - Social comparison admission: 1.1× (mildly undesirable)
  - Digital habits: 1.15× (Andrews et al. 2015 — underreported screen time)
  - (Plus existing 5 categories: HIGH=1.5×, MODERATE-HIGH=1.2×, MODERATE=1.0×, LOW=0.5×, INVERTED=-0.5×)
- **Cross-item reverse-failure tracking**: Per-participant `_participant_reverse_tracking` dict. After ≥2 reverse items, failure rate history influences subsequent reversal probability (0.3 weight blend). Scientific basis: Woods (2006) — trait-like within session.
- **Response pattern validation**: New `_validate_participant_responses()` method with 3 checks:
  - Longstring detection (consecutive identical > 80% of items)
  - IRV (Intra-individual Response Variability) vs engagement mismatch
  - Endpoint utilization vs extremity trait mismatch
- **New condition trait modifiers** for 5 paradigms: narrative transportation, social comparison, gratitude, moral threat, digital distraction
- **4 new domain calibration entries**: narrative engagement, social comparison, digital wellbeing, moral self

#### Iteration 5: Documentation Overhaul
- **Front-facing methods document** (methods_summary.md): Added 3 new domain categories (Narrative & Communication, Digital & Technology, Positive Psychology), behavioral coherence pipeline section, reverse-coded item modeling section, response validation layer section, 10 new scientific references
- **CHANGELOG**: Full entry for v1.0.4.9 with all 5 iterations
- **README**: Updated version and features section
- **All files version-synced to 1.0.4.9**

#### Version Updates
- All files synchronized to v1.0.4.9: app.py, __init__.py, enhanced_simulation_engine.py, qsf_preview.py, response_library.py, persona_library.py, README.md

---

## 2026-02-11 — v1.0.4.6
### Pipeline Quality Overhaul: Domain-Aware Routing, Persona Expansion, Cross-DV Coherence (3 Iterations)

**Core Architecture Change**: `self.detected_domains` (computed via 5-phase domain detection at init) is now the PRIMARY routing signal for all major pipeline methods. Previously, 4 of 5 methods ignored detection results and re-did keyword matching independently.

#### Iteration 1: Domain-Aware Routing Foundation (Steps 1, 2, 3, 6)
- **Step 1 — Domain-aware STEP 2 routing**: Added `_DOMAIN_RELEVANCE` mapping (18 domains → persona domain strings) and `_domain_is_relevant()` gate. Each STEP 2 domain block now checks if the study's detected domains overlap before applying effects. Prevents cross-domain keyword stacking (e.g., health keywords firing in a political study).
- **Step 6 — Effect stacking guard**: After STEP 2, if cumulative effect exceeds ±0.30 and no detected domains match, attenuates by 0.5× and caps to ±0.50. Prevents runaway effects from unrelated domains firing simultaneously.
- **Step 2 — Domain-aware trait modifiers**: Rewrote `_get_condition_trait_modifier()` study-level priming to use `self.detected_domains` directly. 10+ domain-specific priming blocks (political, economic, clinical, organizational, consumer, media). Fallback to keyword matching only when no domains detected.
- **Step 3 — Domain-aware persona weight adjustment**: Rewrote `_adjust_persona_weights_for_study()` to use `_DOMAIN_TO_CATEGORY_BOOST` mapping from detected domains. 14 domain→category mappings with 1.2-1.4× boost multipliers.

#### Iteration 2: Calibration, Persona Expansion, Context-Enriched Selection (Steps 4, 5, 7, 8)
- **Step 8 — Domain-aware STEP 4 scaling (CRITICAL BUG FIX)**: The `_domain_d_multiplier` keyword fallback chain had a bug where `elif` blocks (Health through Punishment, ~13 domains) were siblings of the `if not _used_detected_scaling:` guard instead of nested inside it. When detected-domain routing set the flag, the keyword blocks could OVERWRITE the detected multiplier. Fixed by nesting all keyword blocks inside the guard.
- **Step 4 — Domain-aware response calibration**: Added detected-domain priors at top of `_get_domain_response_calibration()`. Additive baseline adjustments (variance, positivity bias) based on study domain. These complement (don't replace) variable-specific calibrations.
- **Step 5 — Persona domain mapping expansion**: Expanded `applicable_domains` for 10+ existing personas (social_comparer → +consumer/political/economic, prosocial_individual → +economic_games/trust, individualist → +behavioral_economics/consumer/political, loss_averse → +health/consumer, etc.). Fixed duplicate `competitive_achiever` definition. Added 8 new personas:
  - `partisan_ideologue`: Strong partisan with ideological consistency (Iyengar & Westwood 2015)
  - `pragmatic_moderate`: Centrist with low partisanship (Fiorina et al. 2005)
  - `reciprocal_cooperator`: Conditional cooperator matching others' behavior (Fischbacher et al. 2001)
  - `free_rider`: Selfish maximizer in collective action problems (Fischbacher et al. 2001)
  - `fairness_enforcer`: Altruistic punisher of norm violators (Fehr & Gächter 2002)
  - `ingroup_favorer`: Strong ingroup identification and favoritism (Balliet et al. 2014)
  - `egalitarian`: Low SDO, committed to cross-group fairness (Pratto et al. 1994)
- **Step 7 — Study-context-enriched persona selection**: Added condition-text analysis with `_CONDITION_PERSONA_AFFINITIES` mapping (7 keyword groups → persona fragment matches). Study title/description paradigm recognition (dictator game, trust game, political polarization) triggers 1.25× persona boosting.

#### Iteration 3: Validation and Cross-DV Coherence (Steps 9, 10)
- **Step 9 — Persona pool validation**: After initial domain filtering, validates that ≥3 non-response-style personas are available. If pool is too small, expands to `_ADJACENT_DOMAINS` mapping (13 domain → 3 adjacent domain lists). Logs when expansion occurs.
- **Step 10 — Cross-DV coherence**: Per-participant response history tracking (running mean of normalized responses). `_generate_scale_response()` pulls adjusted_tendency toward participant's running average with weight proportional to response consistency (0.02-0.10). Creates realistic within-person coherence beyond g-factor and latent scores. Scientific basis: Podsakoff et al. (2003) CMV accounts for r ≈ 0.10-0.20 shared variance.

#### Version Updates
- All files synchronized to v1.0.4.6: app.py, __init__.py, enhanced_simulation_engine.py, qsf_preview.py, response_library.py, persona_library.py, README.md

---

## 2026-02-11 — v1.0.4.5
### Bug Fixes + Simulation Realism Improvements (3 Iterations)

#### Critical Bug Fix
- **Fixed `NameError: name 'condition_lower' is not defined`** in `_generate_scale_response()`: Variables `condition_lower` and `variable_lower` were used at line ~5846 (domain-persona sensitivity factor) without being defined in the method scope. Added local definitions. This crash occurred during any simulation run with condition effects.
- **Added safety fallback for `selected` variable** on Design page: Defensive check ensures `selected` is always defined before combining with `custom_conditions`, preventing potential NameError on page transitions.
- **Fixed builder state cleanup**: `_finalize_builder_design()` now clears stale `_br_scale_version`, `_br_oe_version` keys and sets `_builder_oe_context_complete`, preventing mixed builder/QSF UI state.

#### Iteration 1: STEP 2 Domain Expansion + Persona × Condition Interactions
- **Expanded 7 thin domains** with 25+ new keyword→effect rules:
  - Domain 6 (Health/Risk): +3 rules — optimistic bias (Weinstein 1980), health literacy (Berkman et al. 2011), descriptive health norms (Cialdini 2003)
  - Domain 7 (Organizational): +3 rules — leader-member exchange (Gerstner & Day 1997), psychological safety (Edmondson 1999), organizational trust (Dirks & Ferrin 2002)
  - Domain 10 (Communication): +4 rules — sleeper effect (Kumkale & Albarracin 2004), elaboration likelihood routes (Petty & Cacioppo 1986), repeated message exposure (Zajonc 1968), source attractiveness (Eagly & Chaiken 1993)
  - Domain 11 (Learning): +3 rules — transfer-appropriate processing (Morris et al. 1977), encoding specificity (Tulving & Thomson 1973), self-reference effect (Symons & Johnson 1997)
  - Domain 12 (Social Identity): +4 rules — recategorization (Gaertner & Dovidio 2000), crossed categorization (Crisp & Hewstone 2007), relative deprivation (Smith et al. 2012), perspective-taking (Galinsky & Moskowitz 2000)
  - Domain 14 (Environmental): +3 rules — noise effects (Szalma & Hancock 2011), color psychology (Mehta & Zhu 2009), music/sound (Kämpfe et al. 2011)
  - Domain 17 (Deception): +3 rules — ethical fading (Tenbrunsel & Messick 2004), incrementalism/slippery slope (Welsh et al. 2015), self-concept maintenance (Mazar et al. 2008)
  - Domain 18 (Power/Status): +5 rules — dominance vs prestige (Cheng et al. 2013), status anxiety (Wilkinson & Pickett 2009), hierarchy legitimacy (Tyler 2006; Jost & Banaji 1994), power and perspective-taking (Galinsky et al. 2006), resource scarcity (Shah et al. 2012)
- **5 new persona × condition interactions** in `_generate_scale_response()`:
  - Political identity × cooperation tendency in economic games (Dimant 2024)
  - Authority × Need for Cognition — low NFC amplifies authority (Petty & Cacioppo 1986)
  - Loss frame × loss aversion trait (Kahneman & Tversky 1979)
  - Stereotype threat × self-efficacy (Steele & Aronson 1995)
  - Environmental concern × green messaging (Stern et al. 1999 — already existed, verified)

#### Iteration 2: Reverse-Item Modeling + SD Domain Sensitivity
- **Enhanced reverse-item modeling** with engagement-differential failure rates:
  - Careless respondents (engagement < 0.35): reversal probability reduced 30%
  - Satisficers (engagement 0.35-0.55): reversal probability reduced 12%
  - Acquiescence pull is 50% STRONGER when reversal fails vs succeeds
  - Scientific basis: Krosnick (1991) satisficing theory, Woods (2006) reversal failure rates
- **Economic game SD sensitivity**: Dictator/trust/ultimatum game DVs now get MODERATE-HIGH SD sensitivity (1.3×) instead of LOW (0.5×). Allocation decisions reveal character → fairness norms create SD pressure. Based on Engel (2011) meta-analysis.
- **SD × Reverse-item interaction**: When reverse item correctly reversed, SD attenuated 15% (already adjusted). When reversal fails, SD amplified 20% (person fighting their own response style).

#### Iteration 3: Persona Library Expansion + Documentation
- **6 new domain-specific personas** (total: 52+):
  - High-Anxiety Individual (Clark & Watson 1991 tripartite model)
  - Authority-Sensitive Individual (Tyler 2006 procedural justice)
  - Competitive Achiever (Vealey 1986 sport confidence)
  - Securely Attached Individual (Brennan et al. 1998 ECR dimensions)
  - Loss-Averse Saver (Kahneman & Tversky 1979 prospect theory)
  - Persuasion-Resistant Individual (Brehm 1966 reactance theory)
- **Persona library version updated** from 1.0.0 → 1.0.4.5 (was 4 versions behind)
- **Documentation updated**: CHANGELOG, README, CLAUDE.md improvement plan

### Version Synchronization
- v1.0.4.5 across: app.py, utils/__init__.py, qsf_preview.py, response_library.py, persona_library.py, enhanced_simulation_engine.py, README.md

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
