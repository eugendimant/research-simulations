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
Hidden password-protected diagnostics page at `?admin=1`. Shows LLM provider stats, simulation history, session state explorer, system info. Password comes from the `ADMIN_PASSWORD` (or `ADMIN_PASSWORD_SHA256`) secret/env var; with none configured the page stays locked. The analytics dashboard uses `ANALYTICS_DASHBOARD_PASSWORD` the same way.

---

## ABSOLUTE RULE: Version Synchronization — ALL 10 Locations, EVERY Commit

**A version mismatch causes a VISIBLE ERROR BANNER for all users.** The app checks `REQUIRED_UTILS_VERSION == utils.__version__` at startup. If they differ by even one digit, users see a yellow warning bar. **It MUST NEVER happen again.**

### The 10 version locations — ALL must contain the EXACT SAME version string:

| # | File | Location |
|---|------|----------|
| 1 | `simulation_app/app.py` | `REQUIRED_UTILS_VERSION = "X.X.X.X"` (line ~57) |
| 2 | `simulation_app/app.py` | `APP_VERSION = "X.X.X.X"` (line ~192) |
| 3 | `simulation_app/app.py` | `BUILD_ID = "YYYYMMDD-vXXXXX-description"` (line ~58) |
| 4 | `simulation_app/utils/__init__.py` | `__version__ = "X.X.X.X"` (line ~68) |
| 5 | `simulation_app/utils/__init__.py` | `Version: X.X.X.X` in docstring (line ~5) |
| 6 | `simulation_app/utils/qsf_preview.py` | `__version__ = "X.X.X.X"` (line ~36) |
| 7 | `simulation_app/utils/response_library.py` | `__version__ = "X.X.X.X"` (line ~66) |
| 7b | `simulation_app/utils/instructor_report.py` | `__version__ = "X.X.X.X"` (line ~9) |
| 8 | `simulation_app/README.md` | `**Version X.X.X.X**` in header (line ~3) |
| 9 | `simulation_app/README.md` | `## The behavioral engine (vX.X.X.X)` section header |
| 10 | `simulation_app/README.md` | `(Version X.X.X.X)` in the citation block at the bottom |

`scripts/check_version_sync.py` is the authority and runs in CI before the
tests: it checks every row above except `BUILD_ID`. Run it before pushing —
`python3 scripts/check_version_sync.py` — rather than trusting this table,
which has drifted from it before (`instructor_report.py` was missing until
v1.3.0.0 and failed a CI run).

### MANDATORY WORKFLOW — Do this BEFORE every commit:

**Step 1: Determine the new version number.**
- Increment the LAST digit by 1. If it was 9, roll to 0 and increment the digit to its left.
- Examples: `1.0.7.3` → `1.0.7.4`, `1.0.7.9` → `1.0.8.0`, `1.0.9.9` → `1.1.0.0`
- **NEVER use two-digit segments** like `.10`, `.11`. Each segment is a single digit 0-9.

**Step 2: Update ALL 10 locations with the SAME version string.** Never touch one file without the other. The #1 failure mode is updating `utils/__init__.py` without updating `REQUIRED_UTILS_VERSION` in `app.py` (or vice versa). Treat them as a single atomic operation.

**Step 3: Update BUILD_ID** to force Streamlit cache invalidation. Format: `"YYYYMMDD-vXXXXX-short-description"`

**Step 4: Verify** — grep for the old version; it should appear NOWHERE:
```bash
grep -r "OLD_VERSION" simulation_app/ --include="*.py" --include="*.md"
```

### Stale Module Cache Recovery (v1.0.7.7)
The app uses `importlib.reload(utils)` as a safe self-healing mechanism when a mismatch is detected. Warning only appears if reload fails — meaning it's a genuine code-level inconsistency.

**Failed approaches (DO NOT retry):** `sys.modules` purge (caused KeyError crashes with concurrent sessions), warning-only (users saw banner after every deploy).

---

## ABSOLUTE RULE: Import Resilience — One Bad Import Must NEVER Take Down the App

**A single failed top-level import in `app.py` (or anything it transitively imports)
crashes the ENTIRE Streamlit app with a redacted `ImportError` — users see a broken
red page, the whole tool is DOWN.** This is the highest-severity failure class
(P0, production-down). It happened once (`from utils.group_management import ...,
_atomic_write_json` when the two files drifted out of sync across a deploy). It MUST
NEVER happen again.

### Hard rules
1. **NEVER hard-import a `_private` symbol from another module at top level.**
   A `_underscore` name is an internal detail that can be renamed/removed without
   notice; importing it across modules is a landmine. If you must, wrap it:
   ```python
   try:
       from utils.x import _helper
   except ImportError:
       def _helper(...): ...   # local fallback so the app still loads
   ```
   Guarded by `test_app_has_no_toplevel_private_cross_module_imports`.
2. **Run the app-load smoke test before EVERY push.** `python3 -m pytest
   tests/test_app_import_safety.py -q` imports `app.py` exactly as Streamlit does
   and fails the instant any top-level import cannot be resolved. Treat a failure
   as release-blocking. Also run a bare `cd simulation_app && python3 -c "import
   importlib.util,sys; sys.path.insert(0,'.'); …exec app.py…"` if unsure.
3. **Symbols an entry point imports are part of the contract.** If you rename,
   move, or delete a function/class that `app.py` (or any imported module)
   references, update EVERY import site in the SAME commit — never split an
   import and its definition across commits/branches (that creates the exact
   drift that downed the app on a partial deploy).
4. **Prefer public, stable names across module boundaries.** If a helper is used
   by another module, it is part of that module's public API — give it a
   non-underscore name or re-export it intentionally.

### Why redacted errors make this worse
Streamlit redacts the error message on deployed apps ("recorded in the logs"), so
users can't self-diagnose. The crash itself must be prevented — a clearer message
is not enough. When the app is reported down, reproduce the import locally first
(`pytest tests/test_app_import_safety.py`), fix the root cause, and verify with the
smoke test before pushing.

---

## ABSOLUTE RULE: API Keys — Never in the Repository

**Six provider API keys were once committed to this public repository.** They
were readable by anyone, several were auto-revoked by their provider's
secret-scanning partnership with GitHub, and all six had to be rotated. A
public repository keeps a deleted secret in its history forever, so this is not
recoverable by deleting the line. It MUST NEVER happen again.

### Hard rules
1. **A key lives in the deployment and a password manager, nowhere else.**
   Streamlit **Settings → Secrets**, an environment variable, or the
   git-ignored `.streamlit/secrets.toml`. Never a module, a comment, a test
   fixture, a commit message, a PR body or a chat message. Never XOR-encoded,
   base64'd or split across lines — obfuscation is still committing it.
   Guarded by `tests/test_no_secrets_in_repo.py`; never work around that test.
2. **Never remove a provider from the chain without adding a replacement.**
   `BUILTIN_PROVIDER_SECRETS` and the `_builtin_providers` list in
   `llm_response_generator.py` ARE the redundancy. Shortening the chain is
   silent until the next rate limit, when there is nothing left to fall to.
   The sole exception is a provider whose free tier no longer exists (Cerebras
   and Mistral AI, removed in v1.3.0.0): a slot nobody can get a key for is not
   redundancy. The chain is Google (3 models) → Groq (2 models) → SambaNova →
   OpenRouter.
3. **Run "Test providers now" after ANY change to the LLM chain** (`?admin=1`
   → LLM tab). Tests stub the network, so a change can pass CI and still have
   broken every live call. The button sends one real request per configured
   provider and prints OK / Failed / Not configured without ever showing key
   material.
4. **A missing key is never an error.** With none configured the built-in
   engine writes the open-ended text, the run completes, numeric data is
   identical, and the user sees a notice — not a banner. Turning a missing key
   into a failure is a bug: `tests/test_free_path_never_errors.py` and
   `test_no_keys_reports_not_configured_not_unreachable` hold that line.
5. **Docs and chain must agree.** `docs/PROVIDER_SETUP.md` lists the providers
   in the order the chain tries them; `tests/test_provider_docs_match_chain.py`
   fails if either side moves alone. Adding a provider means adding its setup
   step and its row in that test's `_SLOT_DOC_NAMES`.

Full policy, with the test that enforces each rule: `docs/KEY_POLICY.md`.
Setup walkthrough for all four providers: `docs/PROVIDER_SETUP.md`.

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
2. Verify version numbers are synchronized (all 10 locations)
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
3. **STEP 2 — Domain-specific semantic effects** (43 domains): Each domain has keyword→effect mappings grounded in literature.
4. **STEP 3 — Stable-hash jitter**: an MD5-derived nudge of ±0.04 so same-meaning condition labels still differ slightly (never positional). Condition trait modifiers — political identity → extremity/consistency, outgroup → negative acquiescence — are a *separate* method, `_get_condition_trait_modifier()` (`enhanced_simulation_engine.py:7571`), applied as STEP 1 of `_generate_scale_response()`.
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

## Effect Fidelity and Text Safety (v1.2.9.1) — DO NOT regress

- **Calibration contract:** a configured Cohen's d is the target on the scale MEAN (single item for one-item scales). `_explicit_effect_scale()` carries the corrections: the composite factor (`_EFFECT_ITEM_RHO = 0.20`), x1.10 for scales of 50+ points, and the v1.2.9.1 empirical terms (divide by `1 + 0.14 ln(min(k, 30) / 3)` for k > 3 items; x1.30 on 2-point and x1.07 on 3-point scales). They were fitted on 12 independent seeds per cell at N = 1,200. With two or more scales the effect is applied to the finished item answers instead (`_apply_user_effect_to_scale`, sized from the scale's realised within-condition SD, randomised rounding, the realised gap between arms is NOT forced), because the cross-scale latent term otherwise cut the realised d to 0.2-0.5 of the request; a lone scale is bit-identical to the old route. **Never calibrate on a single seed**: one seed moves the realised d by about 0.1 at N = 2,400, which looks like a systematic bias. Measured on the merged tree (12 seeds, N = 1,200, 18 scale shapes): 0.88-1.05 of the request, weakest the 4-item 5-point scale, where the alpha-targeting step `_attenuate_inter_item_correlation` adds item noise (1.07 with it switched off). Guards: `tests/test_effect_size_recovery.py`, `tests/test_effect_fidelity_v1291.py`.
- **`auto_effects=False` is a true null.** `_compute_effect_for_condition` returns 0 for conditions without a user effect, and `_compute_condition_trait_modifier` skips every name-based modifier. Name-based trait modifiers are also skipped for conditions named by a user-specified effect (`_is_explicit_condition`).
- **Condition-name matching uses `_kw_hit` / `_word_in` (non-word-character boundaries, `(?<!\w)...(?!\w)`), never substring `in`** ("ai" matched "wait", "low" matched "follow-up"; labels such as `80%`, `$10` or `Treatment (high)` must match: a `\b` boundary silently dropped the requested effect for 9% of corpus labels). Spec variables match on whole words ("Trust" must not reach "Distrust"); a spec that reaches nothing is reported in `effect_sizes_applied.specs` and `generation_warnings`.
- **Game DVs** (SocSim) overwrite item columns. `_reapply_user_effects_after_game_model` restores the user's effect afterwards (randomised rounding, so narrow integer scales are not stuck on whole-point jumps), `_reconcile_composites` keeps every `<Scale>_mean` consistent with its items, and `effect_sizes_observed` / `effect_sizes_applied` are recomputed after that, at the very end of `generate()`.
- **Straight-line handling:** every pass that edits constant rows needs five response options (`_MIN_OPTIONS_FOR_STRAIGHTLINE_LOGIC`): the audit (CHECK 3), the registry-calibrated identical-answer pass (`_apply_identical_answer_realism`, whose shares were measured on 5- to 9-point instruments) and the HBS validator (it skips items with fewer than five observed response options, and it pools numbered items across scales, so a 3-item 7-point scale can still be repaired above its benchmark rate). On binary/3-point scales these passes randomised honest data and erased the condition effect (a requested d of 0.8 came back as 0.61 on a 3-item binary scale). The audit additionally needs five or more scale items OR a block of three or more items: the identical-answer pass was calibrated on audited data and restores the measured share after it, so on a 3- or 4-item block switching the audit off shifted `P(7) - P(6)` by 3.5 points (+0.025 against -0.010 on main) and failed the ceiling-spike guard. Guard: `tests/test_merge_interplay_v1304.py`.
- **Generated text is edited only at grammatical positions**, through `utils/text_cleanup.py`. Never insert, drop or swap words at random positions, never substring-replace words without word boundaries, never append counters like "(2)". Applies to `_apply_deep_variation`, the offline tic/filler code, the stylometric engine and the validator. `tests/test_quality_v1291.py` is the guard.
- **Numeric text boxes** (Qualtrics validation `content_type` / `number_min` / `number_max`, or the question wording) get numbers via `infer_numeric_answer_spec` / `draw_numeric_answer`; a text box that duplicates a numeric DV is dropped (`_drop_oe_duplicating_dvs`).
- **QSF collection** is opt-in per file (`_collect_qsf_if_consented`). `utils/github_qsf_collector.py` validates the payload, caps size, rate-limits uploads and can target a branch other than the deployed one (`GITHUB_QSF_BRANCH`): every upload is a commit, and a commit to the deployed branch redeploys the app.

---

## Within-subjects and mixed designs (v1.3.0.6) — DO NOT regress

- **Between-subjects output is bit-identical.** `design=None` (the default) never touches the new code. `generate()` dispatches to
  `utils/within_design.generate_repeated` only when `engine.design_spec` is set. Guard: `BASELINE_HASHES` in
  `tests/test_within_design_v1306.py` (df and metadata hashes recorded on commit 6d97a95). Never "improve" the between path while
  working on repeated measures.
- **One inner pass, not k runs.** A repeated-measures run builds an inner single-pass engine whose scales are replicated once per
  within-condition (`<DV>_occ<j>`) with a Kronecker correlation matrix; that gives one persona / demographics / attention check /
  careless style per person for free. Columns are renamed to `<DV>_<condition>_<i>` / `_mean`; never generate the conditions with
  independent engine runs (the same person would not exist across conditions). The inner engine shares the outer engine's
  `llm_generator` (the pre-flight check and the watchdog talk to the outer one).
- **A within `cohens_d` is d_av** (mean difference / average SD of the two conditions); d_z is reported beside it
  (`d_z = d_av / sqrt(2(1 - r))`). Specs on one factor combine by least squares, on different factors add; a mixed design's group
  effect is present after the baseline first level unless the spec carries `at` (the minimal interaction route).
  Guards: `test_recovery_of_d_av_*`, `test_chained_specs_*`, `test_factorial_within_effects_add_per_factor`,
  `test_interaction_can_be_requested_at_one_within_level`.
- **Careless is a property of the person.** Straight-liners (flagged, or constant in half or more of the blocks) are constant in every
  block and receive no effect or correlation shift (the effect is scaled up by the careless share so the sample-level d_av holds).
  Attrition removes the LATER presentation positions. Guards: `test_a_careless_person_is_careless_in_every_condition`,
  `test_attrition_removes_the_later_conditions`.
- **Reports never run between-subjects tests on repeated data.** `instructor_report.is_repeated_design(metadata)` routes the Markdown/HTML
  instructor reports to `within_report` (paired t, RM-ANOVA with Mauchly/GG/HF, mixed ANOVA, Wilcoxon/Friedman; numpy only through
  `within_stats`), and the student summary to `within_report.summary_sections`. Guards: `tests/test_within_report_v1306.py`.
- **The selector is real on both paths.** QSF: `design_type_select` (mirrored in `design_type_choice`); builder: `builder_design_type_input`;
  both feed `design_config` and `_design_config_for_engine`. A QSF with repeated, unrandomized blocks only gets a *suggestion*
  (`within_design.suggest_design_from_qsf`); nothing is switched on by itself. Guard: `tests/test_within_app_v1306.py`.

---

## Open-Text Response Generation Architecture

### Two Separate Systems:
1. **Preview** (`_get_sample_text_response()` in app.py): 5-row preview
2. **Full generation** (`_generate_open_response()` in enhanced_simulation_engine.py): Three-level cascade

### Three-Level Cascade:
1. **LLM Generator** (llm_response_generator.py): 7 free provider entries, in order — Gemini 3.1 Flash Lite → Gemini 2.5 Flash → Gemini 2.5 Flash Lite → Groq GPT-OSS 120B → Groq Qwen3.6 27B → SambaNova Llama 3.3 70B → OpenRouter Mistral Small 3.1. `_builtin_providers` in that file is authoritative
2. **ComprehensiveResponseGenerator** (response_library.py): compositional template engine (opener + intent core + domain elaboration + coda). No Markov chain (the unused `text_generator.py` module that held one was removed in v1.2.8.9)
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
| 4. **OE generation budget** | `enhanced_simulation_engine.py` | Uses `disable_permanently()` | 180s per open-ended question |
| 5. **Global watchdog thread** | `app.py` | Daemon thread checks every 30s — **progress-aware** (v1.2.6.4) | 120s stall (primary); 30 min absolute backstop fires ONLY if also stalled |

> **v1.2.6.4 — Progress-aware watchdog:** Generation is NEVER killed while it is
> still making progress (progress callbacks keep `_last_progress_time` fresh).
> The primary kill condition is a genuine **stall** — no progress for
> `_STALL_TIMEOUT` (120s). The absolute ceiling (`_GLOBAL_GENERATION_TIMEOUT`,
> 30 min) is a backstop that fires ONLY when BOTH the ceiling is exceeded AND
> generation has stalled. This lets large legitimate runs (N=10,000 non-LLM,
> or many LLM open-ended questions) run as long as they keep moving. The old
> elapsed-only 10-min hard cap was removed because it killed active work.

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
- `_report_progress("generating", i, n)` fires EVERY participant **during OE generation** (`enhanced_simulation_engine.py:14170`). The scale-generation loop still fires on an interval — `max(1, min(20, n // 20))`, i.e. every ~5% capped at every 20 (`:13420`)
- `_report_progress("open_ended_question", idx, total)` fires per-OE-question
- UI shows: elapsed time, participant count, live LLM stats (AI count vs template count)
- Post-generation: data source breakdown shown when template fallback was used

### Root Causes That Were Fixed (DO NOT reintroduce)

| Bug | What Happened | Where | Fix |
|-----|---------------|-------|-----|
| Auto-recovery cycle | `is_llm_available` re-enabled dead providers every 20s | `llm_response_generator.py` — `is_llm_available` (~:2671) | `_force_disabled` checked first |
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

- `QSFPreviewParser.EXCLUDED_BLOCK_NAMES` contains 644 literal block names
- `QSFPreviewParser.EXCLUDED_BLOCK_PATTERNS` contains 77 regex patterns
- `_is_excluded_block_name()` checks both
- **Be aggressive with exclusions** — better to exclude too much than pollute conditions
- Common patterns to exclude: `trash_`, `unused_`, `old_`, `test_`, `copy_`, consent, demographics, debrief, attention_check

---

## DV Detection: `_detect_scales()` (11 types)

1. Matrix scales (multi-item Likert)
2. Numbered items (Scale_1, Scale_2)
3. Likert scales (grouped single-choice)
4. Sliders (visual analog)
5. Single-item DVs (standalone ratings)
6. Numeric inputs (WTP, quantities)
7. Constant sum (budget allocation; renormalized to the total)
8. Rank order (valid 1..k permutations)
9. Best-worst — **detected only**, no generation
10. Paired comparison — **detected only**, no generation
11. Hot spot / heatmap — **detected only**, `_generate_heatmap_response` has no callers

`single_choice` is NOT one of them: single-choice items are grouped into
`likert`/`single_item`. Always include `detected_from_qsf: True` flag.

---

## State Persistence

- Page-based rendering keeps state in `st.session_state` directly. The
  `_save_step_state()` / `_restore_step_state()` snapshot pair was **removed in
  v1.4.14** (see the note at `app.py:7022`) — do not reintroduce calls to them.
- `_navigate_to()` mirrors `_widget_persist_keys` to `_p_<key>` so the values of
  widgets that are no longer rendered survive a page switch. That list currently
  holds four keys: `study_title`, `study_description`, `team_name`,
  `team_members_raw` (`app.py:7161`).
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
│   │   ├── text_cleanup.py                # Grammar-safe helpers shared by all OE post-processing
│   │   ├── within_design.py               # within / mixed designs: engine side (wide + long, effects, order, attrition)
│   │   ├── within_stats.py, within_report.py, within_scripts.py   # paired / RM / mixed statistics, report sections, scripts
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
2. Forgetting version sync → always use the 10-location checklist
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
- **v1.2.5.3–v1.2.6.4**: Student-driven UX + correctness wave — manual factor-level input, design-table cell→condition mapping, editable DV type + scale anchors + DV description, mediator/moderator variables with correlation patterns, comprehension checks (correct answer + pass rate), manipulation/attention checks as output columns (condition-aware manip checks, binary attn checks), "Condition X:"/"Group X:"/"Treatment X:" prefix stripping, binary (0/1) DV support, DV-type-preserved-on-add
- **v1.2.6.4 PERFORMANCE**: ~29× speedup for large-N runs — `_get_effect_for_condition` memoized per `(condition, variable)` pair (was recomputing heavy regex matching N×items times) + module-level compiled-regex cache in `_word_in`/`_stem_in`. N=10,000 now generates in ~60s (was ~29 min). Watchdog made progress-aware so legitimate long runs are never killed mid-flight.

### Performance Characteristics (v1.2.6.4)
- **Effect computation is memoized** (`self._effect_cache` keyed on `(condition, variable)`). The cache is per-engine-instance, so it resets each run. NEVER add per-participant randomness inside `_compute_effect_for_condition` — the condition effect is a deterministic mean shift; participant variance is applied separately in `_generate_scale_response`. Adding RNG there would be silently cached and break realism.
- **`_word_in`/`_stem_in` use `_WORD_PATTERN_CACHE`/`_STEM_PATTERN_CACHE`** (module-level compiled-regex caches). When adding new keyword-matching helpers, follow the same compile-once pattern — `re.search(pattern_str, ...)` recompiles every call and is the #1 large-N bottleneck.
- **Non-LLM max N = 10,000** (`MAX_SIMULATED_N`). Verified to complete in ~60s. LLM max stays at `MAX_FREE_LLM_N`.

### Next Targets (v1.0.6.x)
1. Scale type detection expansion (forced choice, semantic differential — neither is detected today)
2. LLM response validation layer (off-topic detection, meta-commentary screening)
3. Authority/NFC persona-level interaction in STEP 3

(Narrative transportation's STEP 2 domain shipped in v1.0.4.9, and matrix
detection already exists. Its `narrative_transportation` template set used to be
unreachable; as of v1.2.8.9 it resolves through `_DOMAIN_TEMPLATE_ALIASES`
(`response_library.py:4673`, consulted at `:8675`). Three other `DOMAIN_TEMPLATES`
keys (`ethical_dilemma`, `gratitude_experience`, `gratitude_intervention`) were
dead until v1.3.0.5 because the alias map is consulted only when `domain.value`
is not itself a key; `_DOMAIN_TEMPLATE_EXTENSIONS` now pools them into `gratitude`
and `moral_dilemma`. All 116 keys are reachable, enforced by
`test_every_domain_template_bank_is_reachable`; see `docs/COVERAGE_ROADMAP.md`
item 12d.)

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
