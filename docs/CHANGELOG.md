
## 2026-06-01 — v1.2.8.4
### Codex review: de-serialize concurrent runs + preserve "and how" topical clauses

Two issues from the chatgpt-codex-connector review on PR #352, both fixed with
regression tests.

- **P2 — Avoid serializing complete simulation runs.** The v1.2.7.8
  `_GLOBAL_RNG_LOCK` (added to make the global-RNG seed/restore safe under
  concurrency) was held across the WHOLE run — including LLM prefill and
  per-participant network calls — so two Streamlit sessions ran strictly one at a
  time. **Resolved by removing the lock entirely:** the engine's numeric draws
  already use per-call `np.random.RandomState` (23 sites, seeded from
  `self.seed`/`participant_seed`), and the *only* remaining global-RNG consumers in
  the generation path — `HBSValidator` (straight-line/coherence perturbations) and
  the LLM backoff jitter — were migrated to per-instance RNGs
  (`HBSValidator(seed=self.seed)`; `LLMResponseGenerator._rng`). The engine no
  longer seeds the process-global RNG, so there is nothing to serialize. **Proof:**
  same-seed output is byte-identical across `PYTHONHASHSEED ∈ {0,1,42,31337,271828}`
  with global seeding *removed* (complete evidence that no global RNG affects data),
  and a new test (`test_v1284_generation_is_not_globally_serialized`) confirms
  multiple threads now execute generation concurrently. This supersedes the
  "conscious trade-off" noted in v1.2.7.9.
- **P2 — Preserve topical clauses beginning with "and how".** `_strip_instruction_tail`
  removed everything after "and why/how", so "Describe your experience with
  depression **and how it affects your work**" lost the meaningful clause (→ topic
  "depression"). Now bare "and why/how" is stripped **only when it dangles at the
  very end** ("…and why?"); an "and how <clause>" that continues is preserved, and
  explicit instructional verb tails ("and explain your reasoning") are still
  stripped. New assertions in `test_v1280_strips_instruction_tail_keeps_conjunctive_topics`.

**Validation:** compileall exit 0; 56 tests pass; cross-process determinism
byte-identical across 5 `PYTHONHASHSEED` values; random N=200 across 10 QSFs
all-pass; version synced (1.2.8.4).

## 2026-06-01 — v1.2.8.3
### UX: one-click "Clear all" for auto-detected open-ended questions

When a QSF is uploaded, the app auto-detects open-ended questions and lists them for
review. A "Remove All" button existed, but only at the **bottom** of the section —
below the full list — so users scrolled past it and removed questions one-by-one via
the per-item ✕. Added a prominent **"🗑️ Clear all N"** button at the **top** of the
section (right of the header), so "clear everything the app detected, then add my
own" is a single click up front. Mirrors the existing remove-all logic (bumps the OE
widget-version, keeps the expander open, shows a confirmation notice); the bottom
"Remove All (N)" button next to "Add Open-Ended Question" is unchanged. App-load
smoke test + full suite pass; version synced (1.2.8.3).

## 2026-06-01 — v1.2.8.2
### PRODUCTION-DOWN FIX: one bad import no longer crashes the whole app

The live app went down with a redacted `ImportError` at `app.py:66`
(`from utils.group_management import GroupManager, APIKeyManager,
_atomic_write_json`). Root cause: a **hard top-level import of a `_private`
cross-module symbol**. The instant `app.py` and `group_management.py` drift out of
sync (partial deploy, stale build, version skew, a rename), that single line takes
down the ENTIRE Streamlit app — and Streamlit redacts the message, so users can't
self-diagnose. Highest-severity class (P0).

- **Fix:** `_atomic_write_json` is now imported **defensively** (`try/except
  ImportError` with a local fallback that writes JSON atomically), so the app
  always loads even against an older/mismatched `group_management`. Verified by
  simulating the broken state (group_management without the helper) → app loads via
  fallback instead of crashing.
- **Permanent safeguards (so it never happens again):**
  - New app-load **smoke test** `tests/test_app_import_safety.py` —
    `test_app_imports_without_error` execs `app.py` exactly as Streamlit does and
    fails on ANY unresolved top-level import; `test_app_survives_missing_private_atomic_helper`
    proves the fallback; `test_app_has_no_toplevel_private_cross_module_imports`
    forbids new `from x import _private` at module scope. Run before every push.
  - New **CLAUDE.md ABSOLUTE RULE "Import Resilience"** + Self-Audit Bug-Class #19:
    never hard-import a `_private` symbol across modules; a symbol an entry point
    imports is part of the contract (rename/move/delete it and all import sites in
    the SAME commit).

> **To restore the live site:** deploy this branch (merge to the branch Streamlit
> Cloud tracks, then reboot the app). The fix makes `app.py` resilient to the
> version skew that caused the outage.

**Validation:** compileall exit 0; app-load smoke test (4) + full suite pass;
version synced (1.2.8.2).

## 2026-06-01 — v1.2.8.1
### Numeric realism: heterogeneous item loadings + narrow-scale bounds fix

Two numeric-side improvements found by a self-audit statistical-realism probe
(validated against Xie et al. 2026) and the scale-generation test suite.

- **Heterogeneous inter-item correlations (Xie et al. structural-uniformity tell).**
  `_inject_inter_item_correlation` mixed every item toward the row mean with the
  SAME weight, so all inter-item correlations clustered at ~one value (range 0.64–
  0.72) — a "looks generated" signature. It now uses per-item mixing weights
  (corr(i,j) ≈ wᵢ·wⱼ), scale-stable (derived from a per-scale `_stable_int_hash`
  seed, applied column-wise so they don't average out) and centered on the uniform
  value so the MEAN correlation — hence Cronbach's alpha — and per-item means/SDs
  (condition effects) are preserved. Result: correlation spread 0.08 → **0.25**
  (range 0.56–0.81), alpha and ingroup>outgroup effect (d≈0.9) unchanged.
- **Narrow-scale (2/3-point) bounds violation (correctness bug).** The
  straight-lining correction in `hbs_validator._correct_straightlining` inferred
  `scale_hi` from the straight-liner's own (constant) row and FORCED it ≥ 5, so a
  2-point item could be perturbed to `2 + 1 = 3` — out of bounds (caught by
  `test_2_point_binary`). It now clamps to each column's FULL observed range, so
  binary/3-point scales stay in `[1, scale_points]`. The engine's own
  anti-straight-line jitter (`_audit_individual_consistency`) had the same class of
  latent bug — it read bounds from a leaked loop variable (the last scale, default
  7) — now fixed with a per-column bounds map.

**New tests:** `test_v1281_inter_item_correlations_are_heterogeneous`,
`test_v1281_narrow_scales_never_exceed_bounds` (+ `test_2_point_binary` now passes).
**Validation:** compileall exit 0; 51 tests (bugfix + e2e + scale-gen) pass;
cross-process determinism byte-identical across PYTHONHASHSEED (incl. a mixed
7-point + binary design); random N=200 across 10 QSFs all-pass; version synced
(1.2.8.1). Bug-class #18 added.

## 2026-06-01 — v1.2.8.0
### Offline open-ended realism: punctuation repair + instruction-free topic extraction

A self-audit scan of the offline (non-LLM) open-ended path at N=200 surfaced two
realism defects — the kind a reviewer would flag as "this data looks generated".
Both are now fixed deterministically (no RNG, so same-seed output stays
byte-identical across processes; the fixes improve a given seed's text, they don't
make it non-reproducible).

- **Mechanical punctuation artifacts (`",,"`, ~14% of responses).** The verbal-tic /
  filler / hedge insertion stages append a comma next to an existing comma. A final
  pure-function `_normalize_punctuation()` pass now repairs `",,"`, space-before-
  punctuation, and `",."` — while PRESERVING intentional realism (`...` ellipsis,
  `!!` emphasis, typos, lowercase "i", word repetition like "like like").
- **Instruction text bleeding into the extracted topic (68% of responses for a
  context-bearing question).** Topic extraction from `question_context` returned the
  instruction, not the subject: "Explain your view of the candidate **and why**" →
  topic "the candidate and why" → "I feel good about the candidate and why is
  straightforward". Now `_strip_instruction_tail()` removes trailing "and why / and
  how / and explain ..." and `_strip_instruction_prefix()` removes leading
  imperatives ("Describe the tax plan" → "the tax plan"; "In your own words,
  describe X" → "X") — conservatively, so genuine conjunctive topics ("crime and
  punishment", "crime ... in modern society") and embedded prepositions ("opini-on")
  are preserved. Added a `\b` anchor so a preposition inside a word can't capture a
  spurious leading "on ...".

**Result (offline, N=200):** `",,"` 29→0, space-before-punct 2→0, "and why" leak
136→0; 100% unique responses, 22–282 char length spread. **New tests:**
`test_v1280_normalize_punctuation_repairs_only_mechanical_glitches`,
`test_v1280_strips_instruction_tail_keeps_conjunctive_topics`,
`test_v1280_offline_oe_has_no_glitches_or_instruction_leak`. **Validation:**
compileall exit 0; 38 tests (bugfix + e2e) pass; cross-process determinism
byte-identical across PYTHONHASHSEED; version synced (1.2.8.0). Bug-class #17 added.

## 2026-06-01 — v1.2.7.9
### Deep adversarial audit: LLM pool-key drift, recoverable latch, reuse reproducibility, bounded memory

A second independent adversarial audit of the concurrency/state surface (after the
v1.2.7.8 fixes) surfaced four genuine defects — all verified against the code, all
now fixed with regression tests. Three hid in non-default paths and produced **no
error**, exactly the profile that slips past testing.

- **H1 (HIGH) — LLM response-pool key drift.** The prefill path and the
  per-participant path built the enriched `question_text` (which seeds the pool
  cache key `md5(question_text[:200]|condition|sentiment)`) from two separately
  maintained blocks. The per-participant block appended `\nCondition:` and
  `\nAdditional context:`; the prefill block did not. Whenever the user filled the
  encouraged Design-page **question-context** field, the two keys diverged within
  the first 200 chars, so **every prefilled pool draw missed** — silently wasting
  the prefill budget and forcing an expensive on-demand LLM call per participant
  (CLAUDE.md anti-pattern #29). Both paths now route through ONE shared helper,
  `_build_enriched_question_text()`, so the keys are byte-identical.
- **M1 (MEDIUM) — `free_tier_exhausted_now` was a one-way latch.** Documented as
  "resets on any success", but `_consecutive_transient_batches` only reset inside
  `_generate_batch` — and once the latch tripped, every caller skipped
  `_generate_batch`, so it could never clear. A single burst of transient
  429/503/timeout abandoned the LLM for the **rest of the run** even after the free
  tier recovered seconds later. Now cleared in `reset_providers()` and at the start
  of each new OE question, so a recovered free tier is genuinely re-used.
- **M2 (MEDIUM) — incomplete reuse reset broke reproducibility.** `_generate_body`
  reset the fallback OE generator (`text_generator`) but not the PRIMARY one
  (`comprehensive_generator`), so calling `generate()` twice on the same engine
  produced **different** OE text on the same seed. Latent under the Streamlit UI
  (which rebuilds the engine each run) but a contract break for SDK/batch/regenerate
  reuse. Added `reset()` to `ComprehensiveResponseGenerator` (and the
  `AdaptiveBehavioralEngineV2` wrapper) and call it each run.
- **L1 (LOW) — unbounded dedup memory at max N.** `_used_responses`/`_used_sentences`
  grew without bound (hundreds of thousands of strings at N=10,000 × many OE
  questions). Now FIFO-bounded via companion `deque`s with **deterministic**
  oldest-first eviction (never `set.pop()` — that would reintroduce
  `PYTHONHASHSEED`-dependent non-determinism).

**Conscious trade-off (documented, not a bug):** the `_GLOBAL_RNG_LOCK` is held
across LLM network I/O, so concurrent users serialize. Removing global seeding
entirely would mean routing ~70 RNG call sites (engine `rng = np.random` ×15 +
`text_generator` bare `random.*` ×58) through injected per-run RNGs — a cross-module
refactor that risks the reproducibility just verified correct. Deferred as its own
staged change; real-world impact is low (free-tier rate limits already serialize
concurrent LLM users; non-LLM N=10,000 finishes in ~60s).

**New regression tests:** `test_h1_prefill_runtime_pool_key_match`,
`test_m1_transient_latch_resets_on_reset_providers`,
`test_m2_reused_engine_is_reproducible_non_llm` (29 in `test_bugfixes_v1264.py`).
**Validation:** compileall exit 0 · 35 tests (bugfix + e2e) pass · cross-process
determinism byte-identical across PYTHONHASHSEED ∈ {0,1,42,31337} · random N=200
across 10 QSFs all-pass · version synced (1.2.7.9). Four new bug-classes (#13–16)
+ the lock trade-off recorded in the Self-Audit Bug-Class Catalog.

## 2026-06-01 — v1.2.7.5–v1.2.7.8
### Codex audit fixes + cross-process reproducibility + concurrency safety

- **v1.2.7.5 (Codex audit):** re-indented `socsim/cli.py` into `main()` (whole-tree
  `compileall` now exits 0); replaced salted `hash()` with SHA-256 `_stable_int_hash`
  in `hbs_engine`/`socsim_adapter`; removed a global-seed side effect in
  `persona_library`; widened int→float columns before float writes in
  `hbs_validator`; preserved `_Base_Generation_Source`/`_Simulation_Method`
  provenance; sanitized a token-prefix echo; documented optional deps.
- **v1.2.7.6:** full **cross-process reproducibility** — same seed → byte-identical
  output across `PYTHONHASHSEED` values (replaced remaining salted hashes in seeding
  contexts with stable ordinal/SHA-256 hashing); established the Self-Audit
  Bug-Class Catalog.
- **v1.2.7.7:** fixed `_Generation_Source` being hardcoded to "Non-LLM" (it
  overwrote real per-participant AI/Template provenance — the reason every prior
  free-LLM run showed "100% Non-LLM"); robust batch-JSON parsing (fenced /
  bracket-less payloads); hoisted an `_ext` UnboundLocalError; put the newest free
  Gemini model in front with transient-vs-hard failover; rejected interrogative
  question text from OE topic extraction (question-leak fix).
- **v1.2.7.8 (Codex P1):** serialized the global-RNG temporary-seeding region with a
  process `RLock`; made `ComprehensiveResponseGenerator`'s uniqueness sets
  per-instance (were class-level, leaking fingerprints across concurrent sessions);
  removed `@st.cache_resource` from the QSF parser (per-parse mutable state). Result:
  concurrent same-seed generation → one identical output.

## 2026-05-31 — v1.2.7.0–v1.2.7.4
### Detection↔generation seam: valid joint/numeric DVs, 5 effect domains, data-validity fixes

Closed the core architectural gap where the parser detected rich DV types but
the engine funneled them all through one numeric-Likert pathway, producing
*silently-invalid* data for the types it claimed to detect.

- **Rank-order DVs** now emit valid 1..k permutations (were independent integers
  with duplicate ranks — 0% valid before).
- **Constant-sum DVs** now sum exactly to the total per row (were independent
  integers — 0% valid before), and are exempted from downstream consistency
  audit / bounds-clipping that previously re-broke the sum 2–7% of the time.
- **Numeric money/WTP/count DVs** are reshaped to realistic right-skewed
  marginals (money: log-normal + ~12% floor spike; counts: gamma) via
  rank-assignment that preserves condition/persona ordering (treatment effects
  survive). Classified on DV-specific text only, with word-boundary cue matching
  and a rating-context guard so neutral rating DVs stay untouched.
- **Scale-bound derivation** fixed: non-1-based Qualtrics choice IDs (e.g.
  14–18, 40–44) normalize to 1..N; fractional (0–0.25) and huge (0–100000)
  slider ranges no longer collapse to a constant.
- **+5 literature-grounded effect domains** (emotion induction/regulation,
  misinformation/illusory-truth, aggression/provocation, negotiation, charitable
  giving), contested effects kept conservative; **Dark Triad** norms wired in.
- **Behavioral DV recovery**: sliders / numeric-entry / essays are recovered as
  DVs/OE (with PII and trash-block exclusions) so behavioural-economics surveys
  simulate their real measures instead of a generic placeholder.
- Validated across all 291 example QSFs (0 crashes), the 10 most-recent student
  QSFs (0 issues), 17 regression tests, and a 2,592-combo effect fuzz; the
  type-aware logic was streamlined into named helpers. See
  `docs/COVERAGE_ROADMAP.md` for the full audit and remaining roadmap.

## 2026-02-13 — v1.0.8.1
### Generation Method Chooser, Real-Time Progress Counter, SocSim Integration

#### 3-Option Generation Method Chooser
- **New pre-generation UI**: Users now choose their simulation method before clicking Generate
- **Option 1: Adaptive Behavioral Engine 3.0** — 225+ domains, census-weighted demographics, stylometric fingerprinting, 5 consistency layers, no API needed
- **Option 2: Built-in AI** — ABE 3.0 behavioral engine + free LLM-powered open-ended text via built-in API keys (Groq, Cerebras, Google AI, OpenRouter)
- **Option 3: Your API Key** — ABE 3.0 behavioral engine + user's own LLM API key with auto-detection for 6 providers and visual format validation
- Each option has an expandable "What does this do?" tooltip with detailed explanation
- Visual selection highlighting with checkmark indicator

#### Real-Time Progress Counter
- Replaced static "This typically takes 15-45 seconds" message with live participant counter
- Shows phase-specific updates: persona assignment, scale generation, OE preparation, participant simulation
- Displays: current/total count, percentage, progress bar, elapsed time, estimated time remaining
- SocSim enrichment phase has its own distinct progress indicator (red theme)
- Completion message shows total time taken

#### SocSim Experimental Engine Integration (v0.16)
- **New module**: `utils/socsim_adapter.py` — bridges between main engine and SocSim
- **Game DV detection**: Automatically detects economic game DVs (dictator, trust, ultimatum, public goods, PD, die roll, gift exchange, stag hunt, risk elicitation, beauty contest, bribery, common pool resource)
- **Evidence-traceable simulation**: Uses Fehr-Schmidt inequity aversion, IRT-based survey responses, softmax choice with bounded rationality
- **4 latent behavioral classes**: self-interested (30%), fairness-minded (25%), reciprocator (25%), high-noise (20%)
- **Condition-aware**: Maps experimental conditions to SocSim topic tags and context features (ingroup/outgroup, anonymity, punishment, norms)
- **Non-destructive enrichment**: Standard simulation runs first; SocSim enriches detected game DVs afterward
- **Post-generation feedback**: Shows which DVs were enriched and which games were simulated

#### API Key Handling Improvements
- Visual format validation feedback when entering API keys (green checkmark + detected provider name)
- Auto-detection for 6 provider key formats: Google AI (AIza...), Groq (gsk_...), Cerebras (csk-...), OpenRouter (sk-or-...), SambaNova (snova-...), OpenAI (sk-...)
- Invalid format warning with red indicator

---

## 2026-02-13 — v1.0.8.0
### Massive OE Template Expansion — 20 Iterations of Comprehensive Improvements

#### Iterations 1-5: Domain Vocabulary Explosion
- **`_get_domain_vocabulary()`** expanded from ~15 to **40+ domain categories**: clinical/mental health, developmental/parenting, personality, sports/athletics, legal/forensic, food/nutrition, communication/persuasion, relationships/attachment, cross-cultural, positive psychology, gender/sexuality, cognitive, neuroscience, human factors/UX, financial psychology, gaming/entertainment, social media, decision science, innovation/creativity, risk/safety, negotiation/bargaining, trust/credibility, workplace behavior, AI ethics/alignment, health disparities
- Each domain provides 8-15 specialized vocabulary words used in template composition

#### Iterations 6-8: Extension, Personalization, and Question-Text Mining
- **`_extend()`** expanded with **50+ new StudyDomain entries** covering clinical, developmental, personality, sports, legal, food, communication, relationship, cognitive, financial, cross-cultural, positive psychology, gender, neuroscience, human factors, innovation, risk, social media, negotiation, gambling, remote work, burnout domains
- **`_personalize_for_question()`** enhanced with **8 new domain condition modifiers** (cognitive, sports, clinical, relationship, communication, food, developmental, technology) — all domain-gated
- **Action verb detection** 15→33 patterns; **Object/target detection** 8→24 groups; **Key phrase patterns** 5→15

#### Iterations 9-11: Careless Response Realism + Intent Detection
- **`_make_careless()`** expanded from 8 to **25 templates** with engagement-aware casing
- **Intent detection** expanded with compound phrasings
- **Intensifiers** 4→8 per sentiment; **SD qualifiers** 3→6

#### Iterations 12-14: TextResponseGenerator Template Banks (persona_library.py)
- All 5 question-type template banks expanded: opinion, explanation, feeling, description, evaluation
- All persona styles expanded: engaged, satisficer, extreme, careless, default
- SD hedges 3→6; Follow-up thoughts 5→8 per sentiment

#### Iterations 15-17: Preview System Sync (app.py)
- Phrase patterns synced 4→11; phrase cap 60→150 chars
- Question intent detection added (6 categories); intent-aware preview cores

#### Iterations 18-20: Extensions, Codas, Final Polish
- General extensions per sentiment 3→6; Codas per intent 3-4→5-7
- All files compile-checked; template banks verified for balance

#### Version Synchronization
- Updated to v1.0.8.0 across all 11 version locations

---

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
### Poe (poe.com) Added as Built-in LLM Provider *(REMOVED in v1.2.1.1 — Poe discontinued free API access Feb 2026)*

- ~~**New built-in provider**: Poe (poe.com) added as 6th provider in the LLM failover chain~~ **REMOVED**
- Replaced by Mistral AI (1B tokens/month free) and SambaNova (free Llama 3.1 70B) as supported providers

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
