# Claude Code Development Guidelines

## GOVERNING PROTOCOL: Simulator Agent Protocol

**Every code task on this project follows: Analyze → Research → Plan → Implement → Validate → Deliver. No exceptions.**

Before writing ANY code, read and follow the full protocol and its reference documents:

- `simulation_app/skills/SKILL.md` — **Master protocol** (complexity assessment, iteration loops, quality gates, delivery checklist)
- `simulation_app/skills/detailed-protocol.md` — Complexity worked examples, git conflict resolution, error handling standards
- `simulation_app/skills/response-generation-pipeline.md` — Subject profile generation, sequential item processing, item-type-specific prompting, context window management
- `simulation_app/skills/template-system.md` — Template architecture, cross-correlation encoding, response library management, variation functions
- `simulation_app/skills/continuous-learning.md` — Auto-archive protocol, train/test calibration, quality tracking, prompt evolution, regression prevention
- `simulation_app/skills/human-likeness-checklist.md` — Comprehensive checklist for simulation realism audits

**The protocol is NOT optional.** Even for trivial tasks, complete Step 0 (problem statement + complexity classification + edge cases + success criteria) before touching code. For COMPLEX tasks, all 5 iteration loops are mandatory.

### Quick Reference: Complexity Levels

| Level | Criteria | Iteration Loops |
|-------|----------|-----------------|
| **TRIVIAL** | Single-file, <20 lines, no new dependencies | 1 |
| **MODERATE** | Multi-file or logic change, clear requirements | 3 |
| **COMPLEX** | Cross-module, new features, architectural decisions | 5 |
| **RESEARCH_NEEDED** | Ambiguous requirements, unknown APIs | **STOP. Ask first.** |

Always present the classification to the user for confirmation before proceeding.

---

## Key Terminology

### Persona Pipeline
The full system for generating realistic participant behavior: domain detection → persona filtering → weight adjustment → assignment → trait generation → response generation. Refers to the complete chain from `detected_domains` through persona selection (`_CONDITION_PERSONA_AFFINITIES`, `_ADJACENT_DOMAINS`) to the 10-step simulation pipeline in `enhanced_simulation_engine.py`.

### Admin Dashboard
Hidden password-protected diagnostics page at `?admin=1`. Shows LLM provider stats, simulation history, session state explorer, system info. Password: "Dimant_Admin" (SHA-256 hashed, updated v1.0.4.8).

---

## ABSOLUTE RULE: Version Synchronization — ALL 9 Locations, EVERY Commit

**A version mismatch causes a VISIBLE ERROR BANNER for all users.** The app checks `REQUIRED_UTILS_VERSION == utils.__version__` at startup. If they differ by even one digit, users see a yellow warning bar. **It MUST NEVER happen again.**

### The 9 version locations — ALL must contain the EXACT SAME version string:

| # | File | Location |
|---|------|----------|
| 1 | `simulation_app/app.py` | `REQUIRED_UTILS_VERSION = "X.X.X.X"` (line ~56) |
| 2 | `simulation_app/app.py` | `APP_VERSION = "X.X.X.X"` (line ~122) |
| 3 | `simulation_app/app.py` | `BUILD_ID = "YYYYMMDD-vXXXXX-description"` (line ~57) |
| 4 | `simulation_app/utils/__init__.py` | `__version__ = "X.X.X.X"` (line ~68) |
| 5 | `simulation_app/utils/__init__.py` | `Version: X.X.X.X` in docstring (line ~5) |
| 6 | `simulation_app/utils/qsf_preview.py` | `__version__ = "X.X.X.X"` (line ~36) |
| 7 | `simulation_app/utils/response_library.py` | `__version__ = "X.X.X.X"` (line ~66) |
| 8 | `simulation_app/README.md` | `**Version X.X.X.X**` in header (line ~3) |
| 9 | `simulation_app/README.md` | `## Features (vX.X.X.X)` section header (line ~22) |

### MANDATORY WORKFLOW — Do this BEFORE every commit:

**Step 1: Determine the new version number.**
- Increment the LAST digit by 1. If it was 9, roll to 0 and increment the digit to its left.
- Examples: `1.0.7.3` → `1.0.7.4`, `1.0.7.9` → `1.0.8.0`, `1.0.9.9` → `1.1.0.0`
- **NEVER use two-digit segments** like `.10`, `.11`. Each segment is a single digit 0-9.

**Step 2: Update ALL 9 locations with the SAME version string.** Never touch one file without the other. The #1 failure mode is updating `utils/__init__.py` without updating `REQUIRED_UTILS_VERSION` in `app.py` (or vice versa). Treat them as a single atomic operation.

**Step 3: Update BUILD_ID** to force Streamlit cache invalidation. Format: `"YYYYMMDD-vXXXXX-short-description"`

**Step 4: Verify** — grep for the old version; it should appear NOWHERE:
```bash
grep -r "OLD_VERSION" simulation_app/ --include="*.py" --include="*.md"
```

### Stale Module Cache Recovery (v1.0.7.7)
The app uses `importlib.reload(utils)` as a safe self-healing mechanism when a mismatch is detected. Warning only appears if reload fails — meaning it's a genuine code-level inconsistency.

**Failed approaches (DO NOT retry):** `sys.modules` purge (caused KeyError crashes with concurrent sessions), warning-only (users saw banner after every deploy).

---

## ABSOLUTE RULE: Page Layout — Next at Top, Scroll at Bottom

- **"Continue to..." button**: TOP of page only, right under the stepper (1-2-3-4). Never at the bottom.
- **"↑ Back to top" scroll link**: EVERY page, ALWAYS at the very bottom. NEVER delete it.
- The stepper bar is visual-only (not clickable). Navigation happens via the visible button.
- Navigation buttons appear only when all required fields on the current step are complete.

---

## MANDATORY: PR Link After Every Change

**EVERY response that involves code changes MUST end with a working, mergeable PR link.**

```
## PR Link
**https://github.com/eugendimant/research-simulations/pull/new/claude/[branch-name]**
```

Never leave changes uncommitted. Never forget the PR link.

---

## Code Quality Standards

### Before Every Commit:
1. Run `python3 -m py_compile <file>` on ALL modified Python files
2. Verify version numbers are synchronized (all 9 locations)
3. Run tests: `python3 -m pytest tests/test_e2e.py -v --tb=short`
4. Test the app loads without version mismatch warning
5. Ensure no syntax errors or import failures

### Code Style:
- Type hints for function parameters and returns
- Docstrings for all public functions
- Follow existing code patterns
- Keep functions focused and single-purpose
- Use `get()` for dict access; handle empty lists, None values, missing keys

---

## Science-Informed Behavioral Realism (CRITICAL)

**ALL simulated behavioral data MUST be consistent with established scientific findings.**

### Effect Detection Pipeline: `_get_automatic_condition_effect()`

Runs in this order:
1. **STEP 0 — Relational/Matching Condition Parsing** (fires FIRST): Detects WHO is matched with WHOM. Political identity detection, ingroup (+0.30) vs outgroup (-0.35 to -0.40). Sets `_handled_by_relational = True` to skip Step 1. Economic game DVs amplify by 1.3×.
2. **STEP 1 — Simple valence keywords** (ONLY if STEP 0 didn't handle): "positive", "negative", "reward", "punishment". Note: 'lover' and 'hater' are EXCLUDED (identity markers, not valence).
3. **STEP 2 — Domain-specific semantic effects** (14+ domains): Each domain has keyword→effect mappings grounded in literature.
4. **STEP 3 — Condition trait modifiers**: Political identity → increased extremity/consistency. Outgroup → negative acquiescence bias.
5. **STEP 4 — Domain-aware effect magnitude scaling**: Political + economic game: 1.6×. Political only: 1.3×. Economic game only: 1.2×.

### Economic Game DV Calibration
- **Dictator game**: mean ~28% (Engel 2011 meta-analysis)
- **Trust game**: baseline ~50% (Berg et al. 1995)
- **Ultimatum game**: offers ~40-50%
- **Public goods game**: contributions ~40-60%

### Anti-Patterns for Behavioral Realism
1. Using simple valence keywords for identity conditions
2. Equal effects across ingroup/outgroup — intergroup studies MUST show discrimination
3. Generic 50% baselines for economic games
4. Ignoring domain when scaling effects
5. Treating condition labels literally instead of parsing relational meaning

---

## Open-Text Response Generation Architecture

### Two Separate Systems:
1. **Preview** (`_get_sample_text_response()` in app.py): 5-row preview
2. **Full generation** (`_generate_open_response()` in enhanced_simulation_engine.py): Three-level cascade

### Three-Level Cascade:
1. **LLM Generator** (llm_response_generator.py): Free LLM APIs (Google AI Flash → Lite → Groq → Cerebras → SambaNova → Mistral → OpenRouter)
2. **ComprehensiveResponseGenerator** (response_library.py): Template-based + Markov chain
3. **TextResponseGenerator** (persona_library.py): Basic template fallback

### Key Principle: NO response should EVER be off-topic
- High quality: Detailed, specific, directly addresses question topic
- Medium: Brief but still about the topic
- Low: Very short but topic-relevant ("trump is ok i guess")
- Very low: Gibberish but topic words when possible
- Careless: Short and lazy, but STILL about the actual topic

### Topic Extraction Fallback Chain (every level must extract topic):
1. `question_context` (user-provided on Design page)
2. `question_text` (the actual question)
3. `question_name` / variable name
4. `study_domain`
5. **Last resort**: `"the questions asked"` — NEVER `"this topic"`, `"it"`, `"this"`, or bare pronouns

### Stop-word pattern (used across all files):
```python
_stop = {'this', 'that', 'about', 'what', 'your', 'please', 'describe',
         'explain', 'question', 'context', 'study', 'topic', 'condition',
         'think', 'feel', 'have', 'some', 'with', 'from', 'very', 'really'}
```

### Behavioral Coherence Pipeline (v1.0.4.8+)
Every simulated participant is ONE person. Their numeric responses and open-text must tell a coherent story. `_build_behavioral_profile()` computes response_pattern, intensity, consistency_score, straight_lined flag from numeric responses. This profile flows through all three generator levels. Straight-liners get truncated responses; positive-raters don't get negative text; extreme raters get intensifier phrases.

### Cross-Response Consistency (v1.0.5.0+)
`_participant_voice_memory` tracks each participant's established tone and prior response excerpts across multiple OE questions. Same participant sounds the same across all their OE answers.

---

## CRITICAL: LLM Generation Anti-Hang Architecture (v1.1.1.0+)

**Historical Bug (v1.1.0.9):** LLM generation could hang for 12+ hours even with available API tokens. Root causes were auto-recovery defeating budget enforcement, quality filter silently rejecting valid LLM responses, rate limiter timestamp corruption, and insufficient prefill coverage.

### Five-Layer Defense System (ALL must remain intact)

| Layer | File | Mechanism | Threshold |
|-------|------|-----------|-----------|
| 1. **Permanent disable** | `llm_response_generator.py` | `_force_disabled` flag + `disable_permanently()` — auto-recovery CANNOT undo | N/A |
| 2. **Cumulative failures** | `llm_response_generator.py` | `_cumulative_failure_count` (never resets on success) | 15 total |
| 3. **Per-participant timeout** | `enhanced_simulation_engine.py` | Tracks wall-clock time per OE response | 3 consecutive > 45s |
| 4. **OE generation budget** | `enhanced_simulation_engine.py` | Uses `disable_permanently()` | 180s total |
| 5. **Global watchdog thread** | `app.py` | Daemon thread checks every 30s | 10 min total or 120s stall |

### Anti-Hang Rules (NEVER violate)

1. **NEVER set `_api_available = False` directly** to disable LLM — use `disable_permanently()` instead. Direct assignment can be undone by auto-recovery.
2. **NEVER re-enable LLM after `_force_disabled = True`** within the same generation run. The flag is intentionally irrecoverable.
3. **NEVER remove the quality filter fallback** in `_generate_batch()` — when ALL responses fail quality check, the batch MUST accept them instead of silently discarding.
4. **NEVER reduce the prefill budget below 60s** — insufficient prefill forces expensive on-demand calls for every participant.
5. **NEVER increase batch retry sizes beyond 2 consecutive** — if first 2 batch sizes fail, remaining will too.
6. **`reset_providers()` MUST be a no-op when force-disabled** — prevents infinite retry cycles.

### Pre-Flight Health Check (app.py)
Before generation starts, `engine.llm_generator.health_check(timeout=12)` tests one provider. If it fails, the user sees 3 choices IMMEDIATELY (retry / own API key / template). The user is NEVER left waiting for a dead API.

### Progress Callback Architecture
- `_report_progress("generating", i, n)` fires EVERY participant (not every 5%)
- `_report_progress("open_ended_question", idx, total)` fires per-OE-question
- UI shows: elapsed time, participant count, live LLM stats (AI count vs template count)
- Post-generation: data source breakdown shown when template fallback was used

### Root Causes That Were Fixed (DO NOT reintroduce)

| Bug | What Happened | Where | Fix |
|-----|---------------|-------|-----|
| Auto-recovery cycle | `is_llm_available` re-enabled dead providers every 20s | `llm_response_generator.py:2296` | `_force_disabled` checked first |
| Quality filter too strict | Topic keyword matching rejected valid LLM responses silently | `_is_low_quality_response()` | 3-char prefix matching + accept-on-full-rejection |
| Rate limiter timestamp | `wait_if_needed()` returned without appending timestamp when sleep > 15s | `_RateLimiter.wait_if_needed()` | Returns bool; caller checks |
| Prefill budget too short | 30s filled only 2/15 pool buckets → 500+ on-demand calls | `enhanced_simulation_engine.py` | Increased to 90s |
| Batch retry waste | 4 sizes tried even when first 2 failed | `generate()` in llm_response_generator | Break after 2 consecutive empty |
| `_oe_budget_switched_count` | Never incremented → always reported 0 template fallbacks | OE loop | Incremented on budget exceed and empty response |
| Progress only every 5% | Users saw stale progress for 30+ seconds during OE | OE loop | Every participant now |
| Pool draw → on-demand waste | Pool response failed quality → triggered expensive on-demand | `generate()` pool draw path | Accept pool responses (LLM-generated) |

---

## Streamlit DOM & Navigation (Hard-Won Lessons)

### What Works:
- **Visible `st.button()` at top of each page** — simplest, most reliable. No JS, no iframes.
- **`st.container()` as deferred-render placeholder** — create at top, populate at bottom after widgets set state.
- **Read widget keys directly** — `st.session_state.get(widget_key)` gives CURRENT value, unlike shadow keys.

### What NEVER Works (DO NOT retry):
- `st.markdown('<div class="X">')` does NOT wrap subsequent widgets — creates empty styled elements
- `window.parent.location.href = ...` in iframe JS — destroys WebSocket, loses ALL session state
- Hidden buttons + MutationObserver — container shows as gray bar, buttons leak
- CSS wrapper hiding — Streamlit renders buttons as siblings, not children
- `sys.modules` purge for version cache — KeyError crashes with concurrent sessions

### Builder vs QSF path divergence:
- Builder sets `_skip_qsf_design = True`, skips QSF widgets. Readiness criteria MUST differ per path.
- NEVER assume both paths use the same session_state keys.

### Execution order gotcha:
- Widgets below set state AFTER code above reads it. Fix: `st.container()` placeholder at top, fill at bottom.

---

## Trash/Unused Block Handling

- `EXCLUDED_BLOCK_NAMES` contains 200+ patterns
- `EXCLUDED_BLOCK_PATTERNS` contains regex patterns
- `_is_excluded_block_name()` checks both
- **Be aggressive with exclusions** — better to exclude too much than pollute conditions
- Common patterns to exclude: `trash_`, `unused_`, `old_`, `test_`, `copy_`, consent, demographics, debrief, attention_check

---

## DV Detection: `_detect_scales()` (6 types)

1. Matrix scales (multi-item Likert)
2. Numbered items (Scale_1, Scale_2)
3. Likert scales (grouped single-choice)
4. Sliders (visual analog)
5. Single-item DVs (standalone ratings)
6. Numeric inputs (WTP, quantities)

Always include `detected_from_qsf: True` flag.

---

## State Persistence

- `_save_step_state()` before navigation
- `_restore_step_state()` at step start
- `persist_keys` defines what survives navigation
- Must persist: conditions, factors, confirmed scales/DVs, factorial config, sample/effect size

---

## Project Directory Structure

```
research-simulations/
├── simulation_app/
│   ├── app.py                    # Streamlit entry point
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── enhanced_simulation_engine.py  # 10-step simulation pipeline
│   │   ├── response_library.py            # ComprehensiveResponseGenerator (non-LLM OE)
│   │   ├── persona_library.py             # TextResponseGenerator (fallback OE)
│   │   ├── llm_response_generator.py      # LLM-based OE generation
│   │   ├── qsf_preview.py                # QSF parsing & DV detection
│   │   ├── survey_builder.py
│   │   ├── instructor_report.py
│   │   ├── group_management.py
│   │   ├── schema_validator.py
│   │   └── condition_identifier.py
│   ├── skills/                   # Simulator Agent Protocol
│   │   ├── SKILL.md              # Master protocol
│   │   ├── detailed-protocol.md
│   │   ├── continuous-learning.md
│   │   ├── human-likeness-checklist.md
│   │   ├── response-generation-pipeline.md
│   │   └── template-system.md
│   ├── example_files/
│   └── README.md
├── tests/
│   ├── conftest.py               # Shared path setup & fixtures
│   ├── test_e2e.py               # Main E2E pytest suite
│   └── ...
├── docs/
│   ├── papers/
│   ├── CHANGELOG.md
│   └── *.md
├── CLAUDE.md                     # THIS FILE
├── AGENTS.md
└── MEMORY.md
```

---

## Anti-Patterns (Comprehensive — DO NOT violate)

### Code & Architecture
1. Big-bang rewrites → break into iterations
2. Forgetting version sync → always use the 9-location checklist
3. Assuming state persists → explicitly save and restore
4. Skipping validation → users find edge cases you missed
5. Suppressing exceptions silently (`except Exception: pass`) → always log at minimum

### Streamlit-Specific
6. `st.markdown('<div>')` as wrapper → use `st.container()`
7. `window.parent.location.href` in iframe → destroys session
8. Hidden buttons with JS wiring → use visible `st.button()`
9. "Next" buttons at bottom → only at top under stepper
10. Reading session_state at top for values set below → deferred container pattern
11. Removing scroll buttons when editing → NEVER remove "Back to top"
12. Assuming QSF and builder paths share state keys → they don't

### Open-Text Generation
13. Generic/hardcoded response banks → use question_text, context, condition, study_title
14. Off-topic careless responses ("fine", "ok") → even careless participants write about the TOPIC
15. Consumer/product language defaults ("item", "product") → extract meaningful topics
16. Bare pronouns ('it', 'this') as fallback → multi-level fallback chain
17. Survey meta-commentary ("The survey was well-designed") → responses about the TOPIC, not the survey
18. `{stimulus}` placeholder → use `{topic}` universally
19. Domain-blind extensions → check domain before applying specialization
20. Only auditing primary code paths → generic responses hide in FALLBACK paths

### Behavioral Realism
21. Using valence keywords for identity conditions → parse relational meaning
22. Equal effects across ingroup/outgroup → MUST show discrimination
23. Generic 50% baselines → use published baselines per game type
24. Ignoring domain for effect scaling → political > consumer effect sizes
25. Treating condition labels literally → parse WHO is matched with WHOM

### LLM Generation Pipeline (v1.1.1.0+)
26. Setting `_api_available = False` directly → use `disable_permanently()` (auto-recovery undoes direct sets)
27. Quality filter rejecting ALL batch responses silently → accept when ALL fail (LLM was prompted correctly)
28. Rate limiter `wait_if_needed()` returning void → must return bool so caller can skip
29. Prefill budget < 60s → leaves most pool buckets empty, forces expensive on-demand generation
30. Progress callback every 5% → must be EVERY participant during OE (each can take 30s+)
31. Auto-fallback to templates without user notification → ALWAYS show user what data source was used
32. Batch retry all 4 sizes when first 2 failed → break after 2 consecutive empties
33. Pool draw quality rejection → on-demand generation waste → accept pool responses (they're LLM-generated)
34. Trusting that `_oe_budget_switched_count` tracks itself → must explicitly increment on fallback

---

## Commit Message Format

```
vX.X.X.X: Brief description of changes

- Specific change 1
- Specific change 2

https://claude.ai/code/[session-id]
```

---

## Git Workflow

```
1. git fetch origin && git pull origin main
2. git checkout -b feature/<descriptive-name>
3. Commit atomically with precise messages
4. Before push: git fetch origin main && git rebase origin/main
5. IF conflicts: resolve ALL yourself, re-test, then continue rebase
6. Push ONLY if all tests pass
```

**Files to never touch** (unless explicitly required): README.md, CHANGELOG.md, LICENSE, .gitignore, MEMORY.md.

---

## Improvement History & Current State

### Completed Milestones
- **v1.0.4.5**: Simulation realism — 18 STEP 2 domains, 52+ personas, reverse-item engagement failure, SD domain sensitivity
- **v1.0.4.6**: Pipeline quality — domain-aware routing across all 5 major methods, effect stacking guard, persona pool validation
- **v1.0.4.8**: Behavioral coherence — OE-numeric consistency, `_build_behavioral_profile()`, LLM prompt behavioral hints, cross-correlation enforcement
- **v1.0.5.0**: OE realism — 7-strategy topic extraction, full 7-trait persona integration, 6-check coherence enforcement, participant voice memory
- **v1.1.0.2**: 5x non-LLM OE realism — 8 structural archetypes, domain-specific concrete detail banks (7 domains), natural imperfection engine (10 error types), topic naturalization (pronoun substitution), telltale phrase removal from all phrase banks
- **v1.1.0.3**: 5x non-LLM OE realism round 2 — expanded concrete detail banks (20 domains, 200+ details), trait-driven text modulation (SD hedging, acquiescence, reading speed, consistency), ultra-short response handler, sentence-length variety enforcement, game-subtype vocabulary (dictator/trust/ultimatum/PGG), verbal tic system (10 filler patterns), synonym rotation for cross-participant diversity

### Next Targets (v1.0.6.x)
1. Narrative transportation domain (Green & Brock 2000)
2. Scale type detection expansion (matrix, forced choice, semantic differential)
3. LLM response validation layer (off-topic detection, meta-commentary screening)
4. Authority/NFC persona-level interaction in STEP 3

### Business Roadmap
Phase 1 (Foundation): User accounts + persistent workspaces + billing infrastructure
Phase 2 (Revenue): Paid tiers + REST API + Python SDK + template marketplace
Phase 3 (Scale): LMS integration (Canvas LTI) + Enterprise SSO + R SDK

---

## Scientific References Embedded in Code

| Topic | Reference | Value |
|-------|-----------|-------|
| Intergroup discrimination | Iyengar & Westwood (2015) | Political partisans discriminate in economic games |
| Dictator game baseline | Engel (2011) meta-analysis | Mean giving ~28% |
| Political polarization | Dimant (2024) | d ≈ 0.6-0.9 |
| Intergroup cooperation | Balliet et al. (2014) | Ingroup favoritism |
| Racial discrimination | Fershtman & Gneezy (2001) | Ethnic discrimination in trust/dictator |
| Reverse item failure | Woods (2006) | 10-15% ignore directionality |
| Acquiescence × reverse | Weijters et al. (2010) | +0.5 point error inflation |
| SD domain sensitivity | Nederhof (1985 meta) | d = 0.25-0.75 for sensitive topics |
| Extremity style | Greenleaf (1992) | Response style calibrations |
| SD manifestation | Paulhus (2002) | Impression management vs self-deception |
| Rating-text consistency | Krosnick (1999); Podsakoff et al. (2003) | OE must match numeric |
