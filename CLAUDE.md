# Claude Code Development Guidelines

## ABSOLUTE RULE: Page Layout — Next at Top, Scroll at Bottom

**THESE RULES CANNOT BE OVERRIDDEN UNDER ANY CIRCUMSTANCES.**

### "Back to top" scroll link — EVERY page, ALWAYS at the bottom
- **NEVER EVER delete the "Back to top" scroll link from the bottom of ANY page.**
- Every single wizard page (0, 1, 2, 3) MUST have `↑ Back to top` at the very bottom.
- This is the ONLY element allowed at the bottom. Nothing else.
- If you are editing a page and you see the scroll link, DO NOT TOUCH IT.
- If you are removing other elements, LEAVE THE SCROLL LINK IN PLACE.

### "Continue to..." button — TOP of page only
- NEVER place "Next", "Continue", or any forward-navigation button at the bottom of a page.
- All forward-navigation buttons go at the TOP of the page, right underneath the stepper (1-2-3-4 progress bar).
- Navigation buttons appear only when all required fields on the current step are complete.
- On the Design page, show a visual checklist of what's still needed before the button appears.
- The stepper bar (1-2-3-4) is visual-only (not clickable). Navigation happens via the visible "Continue to..." button at the top.

---

## MANDATORY: PR Link After Every Change

**EVERY response that involves code changes MUST end with a working, mergeable PR link.**

After completing any task:
1. Commit all changes with descriptive message
2. Push to the feature branch
3. Provide the PR link in this format:
   ```
   ## PR Link
   **https://github.com/eugendimant/research-simulations/pull/new/claude/[branch-name]**
   ```

Never leave changes uncommitted. Never forget the PR link.

---

## Version Management (CRITICAL - MUST BE AUTOMATED)

**VERSION SYNCHRONIZATION IS MANDATORY AND MUST BE FULLY AUTOMATED**

When making ANY changes to the codebase, ALL version numbers must be updated together. This includes:

### Files that MUST be updated together:

1. **`simulation_app/app.py`**
   - `REQUIRED_UTILS_VERSION` (line ~46) - MUST match utils `__version__`
   - `APP_VERSION` (line ~100) - Main app version
   - `BUILD_ID` (line ~47) - Change to force cache invalidation

2. **`simulation_app/utils/__init__.py`**
   - `__version__` - Package version (MUST match REQUIRED_UTILS_VERSION)
   - Docstring `Version:` line - MUST match `__version__`

3. **`simulation_app/utils/qsf_preview.py`**
   - `__version__` - Should match package version

4. **`simulation_app/utils/response_library.py`**
   - `__version__` - Should match package version

5. **`simulation_app/README.md`**
   - Version in header line
   - `## Features (vX.X.X)` section header

### Version Update Checklist (DO THIS EVERY TIME):

```
[ ] REQUIRED_UTILS_VERSION in app.py
[ ] APP_VERSION in app.py
[ ] BUILD_ID in app.py
[ ] __version__ in utils/__init__.py
[ ] Version: X.X.X in utils/__init__.py docstring
[ ] __version__ in utils/qsf_preview.py
[ ] __version__ in utils/response_library.py
[ ] Version in README.md header
[ ] Features section header in README.md
```

### Version Numbering Scheme (CRITICAL)

**Each version segment goes from 0 to 9 only. After 9, it rolls back to 0 and the digit to the left increments.**

Examples:
- `1.0.3.9` → next is `1.0.4.0` (NOT `1.0.3.10`)
- `1.0.9.9` → next is `1.1.0.0`
- `1.9.9.9` → next is `2.0.0.0`

**NEVER use two-digit segments like `.10`, `.11`, `.12`.** Each segment is a single digit 0-9.

### Why This Matters:

The app performs a version check at startup:
```python
if hasattr(utils, '__version__') and utils.__version__ != REQUIRED_UTILS_VERSION:
    st.warning(f"Utils version mismatch: expected {REQUIRED_UTILS_VERSION}, got {utils.__version__}")
```

If versions don't match, users see a warning and the app may behave unexpectedly due to Streamlit's module caching.

---

## Code Quality Standards

### Before Every Commit:
1. Run `python3 -m py_compile <file>` on ALL modified Python files
2. Verify version numbers are synchronized (see checklist above)
3. Test the app loads without version mismatch warning
4. Ensure no syntax errors or import failures

### Code Style:
- Use type hints for function parameters and returns
- Include docstrings for all public functions
- Follow existing code patterns in the codebase
- Keep functions focused and single-purpose

### Error Handling:
- Always handle edge cases (empty lists, None values, missing keys)
- Use defensive programming with `get()` for dict access
- Log errors appropriately using the `_log()` method where available

---

## Behavioral Data Simulation Requirements

### Trash/Unused Block Handling

**CRITICAL: Trash blocks must NEVER affect simulation**

- `EXCLUDED_BLOCK_NAMES` in qsf_preview.py contains 200+ patterns
- `EXCLUDED_BLOCK_PATTERNS` contains regex patterns for detection
- `_is_excluded_block_name()` method checks both
- `_get_condition_candidates()` in app.py filters conditions

Always add new exclusion patterns when discovering new trash block naming conventions.

### Condition Detection

Conditions must be properly detected from QSF files:
- Block names that represent experimental conditions
- Embedded data fields used for randomization
- BlockRandomizer elements

**Never include these as conditions:**
- Trash/unused blocks
- Admin blocks (consent, demographics, debrief)
- Structural blocks (intro, instructions, end)
- Quality control blocks (attention checks, manipulation checks)

### DV Detection

The `_detect_scales()` method in qsf_preview.py detects 6 types:
- Matrix scales (multi-item Likert)
- Numbered items (e.g., Scale_1, Scale_2)
- Likert scales (grouped single-choice)
- Sliders (visual analog scales)
- Single-item DVs (standalone rating questions)
- Numeric inputs (willingness to pay, etc.)

Always include `detected_from_qsf: True` flag to distinguish from manual entries.

### Factorial Design

For factorial experiments (2×2, 2×3, etc.):
- Use `_render_factorial_design_table()` for visual setup
- Generate crossed conditions (Factor1_Level × Factor2_Level)
- Save factors to session state for persistence
- Display clear design table with cell numbers

### State Persistence

When users navigate between steps, their selections MUST persist:

- `_save_step_state()` - Call before navigation
- `_restore_step_state()` - Call at step start
- `persist_keys` list defines what to save

Key state that must persist:
- Selected conditions and factors
- Confirmed scales/DVs
- Factorial table configuration
- Sample size and effect size settings

### Science-Informed Behavioral Realism (CRITICAL)

**ALL simulated behavioral data MUST be consistent with established scientific findings.**

The simulation engine must produce results that make sense given the study design, domain, and conditions. Nonsensical results (e.g., equal intergroup giving in a political polarization dictator game) are unacceptable.

#### Architecture: `_get_automatic_condition_effect()` in enhanced_simulation_engine.py

The effect detection pipeline runs in this order:
1. **STEP 0 — Relational/Matching Condition Parsing** (fires FIRST):
   - Detects WHO is matched with WHOM (e.g., "trump lover, trump hater" = outgroup pairing)
   - Political identity detection: figures (trump, biden, obama) + attitudes (lover, hater, supporter, opponent)
   - Ingroup matching → positive effect (+0.30), outgroup matching → negative effect (-0.35 to -0.40)
   - Sets `_handled_by_relational = True` to skip simple valence keywords
   - Economic game DVs amplify intergroup effects by 1.3×
   - General intergroup matching (racial, ethnic, religious, gender identity)

2. **STEP 1 — Simple valence keywords** (ONLY if STEP 0 didn't handle it):
   - Words like "positive", "negative", "reward", "punishment"
   - NOTE: 'lover' and 'hater' are EXCLUDED from valence keywords (they're identity markers, not valence)

3. **STEP 2 — Domain-specific semantic effects** (14+ research domain models):
   - Behavioral economics, social psychology, political science, consumer, health, etc.
   - Each domain has its own keyword → effect mappings grounded in literature

4. **STEP 3 — Condition trait modifiers** (personality-level effects):
   - Political identity conditions → increased extremity, consistency
   - Outgroup conditions → negative acquiescence bias, higher extremity
   - Economic game conditions → increased engagement

5. **STEP 4 — Domain-aware effect magnitude scaling**:
   - Political + economic game: 1.6× multiplier (Dimant 2024: d ≈ 0.6-0.9)
   - Political only: 1.3× multiplier
   - Economic game only: 1.2× multiplier

#### Economic Game DV Calibration (`_get_domain_response_calibration()`)

Economic games have well-established baselines from meta-analyses:
- **Dictator game**: mean_adjustment = -0.22 (centers at ~28%, Engel 2011 meta-analysis)
- **Trust game**: baseline at ~50% (Berg et al. 1995)
- **Ultimatum game**: mean_adjustment = -0.02 (offers ~40-50%)
- **Public goods game**: mean_adjustment = -0.05 (contributions ~40-60%)

#### Key Scientific References Embedded in Code

- **Intergroup discrimination**: Iyengar & Westwood (2015) — political partisans discriminate in economic games
- **Dictator game baseline**: Engel (2011) meta-analysis — mean giving ~28%
- **Political polarization effects**: Dimant (2024) — d ≈ 0.6-0.9 for partisan intergroup effects
- **Intergroup cooperation**: Balliet et al. (2014) — ingroup favoritism in cooperation games
- **Racial discrimination**: Fershtman & Gneezy (2001) — ethnic discrimination in trust/dictator games

#### Anti-Patterns for Behavioral Realism

1. **Using simple valence keywords for identity conditions**: "trump lover" should NOT trigger "lover" as positive valence. It's an identity marker describing WHO the person is, not how they feel.
2. **Equal effects across ingroup/outgroup**: Intergroup studies MUST show discrimination effects. Equal giving/trust across conditions is almost always wrong.
3. **Generic 50% baselines for economic games**: Each game type has established baselines from decades of research. Use them.
4. **Ignoring domain when scaling effects**: Political polarization studies have larger effect sizes than consumer preference studies. Scale accordingly.
5. **Treating condition labels literally**: "trump hater, trump hater" = ingroup (two haters matched together). "trump lover, trump hater" = outgroup (opposing attitudes matched). Parse the RELATIONAL meaning, not individual words.

---

## Domain & Question Type Coverage

### Research Domains (225+)
Organized into 33 categories:
- Behavioral Economics (12)
- Social Psychology (15)
- Political Science (10)
- Consumer/Marketing (10)
- Organizational Behavior (10)
- Technology/AI (10)
- Health Psychology (10)
- And 26 more categories...

### Question Types (40)
- Explanatory: explanation, justification, reasoning, causation, motivation
- Descriptive: description, narration, elaboration, detail, context
- Evaluative: evaluation, assessment, comparison, critique, rating_explanation, judgment, appraisal
- Reflective: reflection, introspection, memory, experience, recall
- Opinion/Attitude: opinion, belief, preference, attitude, value, worldview
- Forward-looking: prediction, intention, suggestion, recommendation, advice
- Associative: association, impression, perception
- Feedback: feedback, comment, observation
- General: general (catch-all)

---

## Project Directory Structure

```
research-simulations/
├── simulation_app/          # Main application code
│   ├── app.py               # Streamlit app entry point
│   ├── utils/               # Core utility modules
│   │   ├── __init__.py
│   │   ├── enhanced_simulation_engine.py
│   │   ├── instructor_report.py
│   │   ├── qsf_preview.py
│   │   ├── survey_builder.py
│   │   ├── response_library.py
│   │   ├── persona_library.py
│   │   ├── group_management.py
│   │   ├── schema_validator.py
│   │   └── condition_identifier.py
│   ├── example_files/       # QSF training data & examples
│   │   ├── *.qsf            # QSF training files
│   │   └── karlijn_case_study/  # Validation case study
│   └── README.md
├── tests/                   # All test files
│   ├── conftest.py          # Shared path setup & fixtures
│   ├── test_e2e.py          # Main E2E pytest suite
│   ├── test_builder_parsing.py
│   └── ...
├── docs/                    # Documentation
│   ├── papers/              # Research papers
│   ├── CHANGELOG.md
│   └── *.md
├── CLAUDE.md                # This file
└── .gitignore
```

## Testing Changes

Before committing:
1. Run `python3 -m py_compile <file>` on modified Python files
2. Verify version numbers are synchronized
3. Run tests: `python3 -m pytest tests/test_e2e.py -v --tb=short`
4. Test the app loads without version mismatch warning
5. Verify trash blocks are properly excluded
6. Check that state persists across step navigation

**Test directory**: All tests live in `tests/`. Path setup is handled by `tests/conftest.py` — individual test files do not need `sys.path` manipulation.

---

## Commit Message Format

Use clear, descriptive commit messages:

```
v2.X.X: Brief description of changes

- Specific change 1
- Specific change 2
- Specific change 3

https://claude.ai/code/[session-id]
```

Always include the Claude session link for traceability.

---

## Development Philosophy & Learnings

### The Power of Iterative Improvement

**CRITICAL: Complex systems improve best through focused iterations, not big-bang rewrites.**

When improving the simulation tool, we learned that requesting "10 iterations of improvements" is far more effective than a single massive change. Each iteration should:

1. **Focus on one area**: Trash handling, then DV detection, then UI, etc.
2. **Build on previous work**: Each iteration benefits from prior improvements
3. **Allow for discovery**: Issues often reveal themselves only after earlier fixes
4. **Maintain stability**: Small changes are easier to test and debug

Example iteration sequence that worked well:
1. Trash block exclusion patterns (foundation)
2. Domain expansion (content)
3. Question type handlers (content)
4. Factorial design table (UI/UX)
5. State persistence (reliability)
6. UI streamlining (polish)
7. DV detection improvements (accuracy)
8. Validation enhancements (robustness)
9. Condition detection filtering (accuracy)
10. Documentation updates (maintenance)

### Key Learnings by Area

#### Trash Block Handling
- **Start with exact matches, then add patterns**: `EXCLUDED_BLOCK_NAMES` handles known names, `EXCLUDED_BLOCK_PATTERNS` catches variations
- **Be aggressive with exclusions**: Better to exclude too much than pollute conditions with trash
- **Common naming conventions to exclude**:
  - Prefixes: `trash_`, `unused_`, `old_`, `test_`, `copy_`, `delete_`, `archive_`
  - Suffixes: `_old`, `_copy`, `_backup`, `_unused`, `_v1`, `_v2`, `_DONOTUSE`
  - Admin blocks: consent, demographics, debrief, instructions, intro, end
  - Quality control: attention_check, manipulation_check, IMC, screener

#### Version Synchronization
- **Learned the hard way**: Mismatched versions cause user-visible warnings
- **Streamlit caches modules**: Old versions persist unless BUILD_ID changes
- **Automate the checklist**: Never rely on memory for multi-file updates
- **Version checks are valuable**: They catch deployment issues early

#### State Persistence
- **Users lose work without it**: Navigating steps should never clear selections
- **Define persist_keys explicitly**: Be deliberate about what survives navigation
- **Save before, restore after**: `_save_step_state()` before navigation, `_restore_step_state()` at step start
- **Test navigation thoroughly**: Forward and backward, direct step clicks, edge cases

#### DV Detection
- **QSF files vary widely**: Support multiple detection strategies
- **6 detection types cover most cases**:
  1. Matrix scales (multi-item Likert)
  2. Numbered items (Scale_1, Scale_2, etc.)
  3. Likert scales (grouped single-choice)
  4. Sliders (visual analog)
  5. Single-item DVs (standalone ratings)
  6. Numeric inputs (WTP, quantities)
- **Flag detection source**: `detected_from_qsf: True` distinguishes auto from manual
- **Include question text**: Helps users verify correct detection

#### Factorial Design
- **Visual tables reduce errors**: Users understand 2×2 grids better than lists
- **Auto-generate crossed conditions**: Factor1_Level1 × Factor2_Level1, etc.
- **Persist factorial state**: Factors, levels, and crossed conditions must survive navigation
- **Clear cell numbering**: "Cell 1", "Cell 2" helps with analysis planning

#### UI/UX Streamlining
- **Scroll to top on navigation**: Users expect to start at the beginning of each step
- **Type badges for DVs**: "Matrix", "Slider", etc. help users understand what was detected
- **Easy remove buttons**: One-click removal of incorrect detections
- **Progressive disclosure**: Show advanced options only when needed
- **Confirmation banners for checkboxes**: Plain checkboxes are too subtle — add amber `.confirm-banner.pending` before unchecked items and green `.confirm-banner.done` after checked items. This makes action items visually prominent.
- **Back buttons on every page (except first)**: Users need to go back to fix things. Place Back button in left column, Continue in right column, at the top under the stepper.
- **No scroll button on Setup page**: Setup is too short to need a scroll button. Only pages with substantial content need "Back to top".
- **Stepper subtitles should say "Complete" not truncated titles**: Truncated study titles look unprofessional. Use simple "Complete" status text instead.

#### Condition Detection
- **Comprehensive filtering is essential**: Apply all exclusion patterns before presenting candidates
- **Multiple sources**: Block names, embedded data, BlockRandomizer elements
- **Never auto-select trash**: Always filter before presenting to user
- **Validate against known patterns**: Check both exact matches and regex patterns

#### Streamlit DOM & Navigation (CRITICAL — Hard-Won Lessons)

**`st.markdown('<div class="X">')` does NOT wrap subsequent widgets.**
- Streamlit renders each widget independently as siblings in the DOM.
- A `<div>` opened in one `st.markdown()` call is auto-closed by the browser. The next `st.button()`, `st.text_input()`, etc. render as siblings, NOT children.
- **NEVER** use `st.markdown('<div>')` to create wrapper containers for styling. It creates empty styled elements (e.g., empty white cards).
- **Use `st.container()`** if you need a proper wrapping DOM element.

**`st.container(height=N)` creates a clipped container.**
- Creates a `<div>` with `max-height: Npx; overflow: auto;` inline style.
- With `height=1`, contents are clipped to 1px tall — effectively invisible.
- Can be targeted with CSS: `div[data-testid="stVerticalBlockBorderWrapper"][style*="max-height: 1px"]`
- Programmatic `.click()` on buttons inside works even when container is clipped.
- Available since Streamlit 1.29.0. Use `try/except TypeError` for older versions.

**`_st_components.html()` creates an iframe.**
- JS inside can access `window.parent.document` (same-origin sandbox).
- The iframe is CACHED by Streamlit — if the HTML string doesn't change between reruns, the iframe is preserved and JS does NOT re-execute.
- **MutationObserver created in the iframe persists across Streamlit reruns.** This is the correct pattern for continuously re-wiring event handlers.
- **NEVER use `window.parent.location.href = ...`** — it causes a full page reload which destroys the WebSocket connection and loses ALL session state.

**FINAL CORRECT navigation pattern: Visible `st.button()` at top of each page.**
- Stepper (1-2-3-4) is VISUAL ONLY — no click handlers, no JS wiring.
- Each page has a visible "Continue to [Next Step]" primary button right under the stepper.
- Button only appears when the current step's required fields are complete.
- Button calls `_navigate_to(next_page_index)` which sets `_pending_nav` + `st.rerun()`.
- This is the simplest, most reliable approach. No JS, no iframes, no hidden buttons.

**Streamlit execution order: widgets below set state AFTER code above reads it (CRITICAL).**
- When a checklist/nav at the TOP of a page reads `st.session_state["some_key"]`, it gets the PREVIOUS rerun's value if the widget that sets `"some_key"` is rendered BELOW.
- On the rerun triggered by a checkbox change, the checklist was already rendered with old values.
- **FIX: Use `st.container()` as a deferred-render placeholder.** Create the container at the top of the page (so it appears visually at the top), but populate it with `with container:` at the BOTTOM of the page after all widgets have set their values. Content appears at the container's visual position but uses CURRENT values.
- **Alternative FIX: Read widget keys directly.** Widget keys (e.g., `dv_confirm_checkbox_v{N}`) are updated by Streamlit BEFORE the script runs. So `st.session_state.get(widget_key)` at the top gives the CURRENT value, unlike shadow keys set by explicit code (e.g., `scales_confirmed`) which are updated later.

**Builder vs QSF path divergence on Design page.**
- The Design page has two paths: builder (conversational) and QSF (file upload).
- The builder path sets `_skip_qsf_design = True` and skips all QSF widgets (including DV/OE confirmation checkboxes).
- Readiness criteria MUST differ per path. Builder readiness = `inferred_design` has ≥2 conditions and ≥1 scale. QSF readiness = conditions + design inferred + scales_confirmed + open_ended_confirmed.
- **NEVER assume both paths use the same session_state keys.** Always check which path is active.

**FAILED approaches (DO NOT retry):**
1. **Hidden buttons + MutationObserver (v1.0.3.4):** Buttons inside `st.container(height=1)` with iframe JS wiring stepper clicks. Failed: container still showed as gray bar, buttons leaked into UI.
2. **Query-param navigation (v1.0.3.3):** `window.parent.location.href = url` causes full page reload → new WebSocket → session state lost.
3. **CSS wrapper hiding (v1.0.3.1-3.2):** `st.markdown('<div class="hidden">')` + buttons. Failed: Streamlit renders buttons as siblings, not children — wrapper is empty, buttons visible.
4. **MutationObserver + setInterval hiding (v1.0.3.2):** JS-based button hiding with 3 defense layers. Failed: timing issues, buttons flash before hiding.

### Anti-Patterns to Avoid

1. **Big-bang rewrites**: Break into iterations instead
2. **Forgetting version sync**: Always use the checklist
3. **Assuming state persists**: Explicitly save and restore
4. **Trusting QSF block names**: Always filter for trash
5. **Hardcoding patterns**: Use configurable lists for exclusions
6. **Skipping validation**: Users will find edge cases you missed
7. **Ignoring UI feedback**: Scroll position, visual feedback matter
8. **Using `st.markdown('<div>')` as wrapper**: It doesn't wrap — use `st.container()` instead
9. **Using `window.parent.location.href` in iframe JS**: Destroys session state
10. **Hidden buttons with JS wiring**: Too complex, unreliable — use visible `st.button()` instead
11. **Putting "Next" buttons at the bottom of pages**: NEVER. Only at the top, right under the stepper. Bottom only has "Back to top" scroll link.
12. **Reading session_state at page top for values set by widgets below**: Use `st.container()` placeholder at top, fill at bottom — or read widget keys directly.
13. **Removing scroll buttons when editing page structure**: NEVER remove "Back to top" scroll links. They must exist at the bottom of every page (except Setup which is too short).
14. **Assuming both QSF and builder paths use the same state keys**: They don't. Always check which path is active before checking readiness.
15. **Using generic/hardcoded response banks for open-text preview**: ALWAYS use question_text, question_context, condition, and study_title to generate topical preview responses. Generic meta-commentary ("the study was interesting") is NEVER acceptable.
16. **Generating off-topic careless responses**: Even careless/low-effort participants write about the TOPIC. "trump is ok i guess" NOT "fine". Extract topic words and build short topic-relevant careless versions.
17. **Defaulting to consumer/product language in templates**: Don't use "item", "product", "aspect" as fallback context values. Extract meaningful topics from question text. Condition modifiers like "AI-recommended" only apply to consumer/AI domains.
18. **Using bare pronouns ('it', 'this') as topic fallbacks**: NEVER use `'it'` or `'this'` when topic extraction fails. Always have a multi-level fallback chain: question_context → question_text → question_name → study_domain → "the questions asked". A response like "it is ok i guess" is a bug.
19. **Suppressing exceptions silently (`except Exception: pass`)**: Always log the error at minimum (`logger.debug`). Silent suppression makes debugging impossible when the cascade falls through to a worse generator.
20. **Writing survey/study meta-commentary in templates**: Templates should be about the TOPIC, not about the survey experience. "The survey was well-designed" is meta-commentary. "I feel strongly about [topic]" is topical. Real participants don't comment on survey design in their open-text responses.
21. **Using {stimulus} placeholder in templates**: `{stimulus}` is legacy consumer language. Use `{topic}` universally. The context dict should map `{topic}` to actual content words extracted from the question text.
22. **Domain-blind extensions and personalizations**: `_extend()`, `_personalize_for_question()`, and condition modifiers MUST be domain-aware. Political studies don't get "About the product" intros. Health studies don't get "AI-recommended" prefixes. Check domain before applying specialization.
23. **Only auditing primary code paths for generic responses**: Generic responses hide in FALLBACK paths. Audit every `else` branch, every `or "default"`, every `if not X:` handler. The primary path may work perfectly while edge cases produce garbage.

#### Open-Text Response Generation (CRITICAL — v1.0.3.8 Overhaul)

**TWO SEPARATE SYSTEMS exist for open-text responses:**
1. **Preview system** (`_get_sample_text_response()` in app.py): Generates 5-row preview data
2. **Full generation system** (`_generate_open_response()` in enhanced_simulation_engine.py): Three-level cascade for actual dataset generation

**The three-level cascade (full generation):**
1. **LLM Generator** (llm_response_generator.py): Uses free LLM APIs (Gemini Flash Lite → Gemma 3 → Groq → Cerebras → OpenRouter) with sophisticated prompts
2. **ComprehensiveResponseGenerator** (response_library.py): Template-based with 225+ domain-specific template sets, Markov chain generation
3. **TextResponseGenerator** (persona_library.py): Basic template-based fallback

**Root causes of broken responses (pre-v1.0.3.8):**
- Preview generator had **hardcoded generic response banks** ("I found this experience engaging", "Made me think about the topic") that completely ignored question text, context, conditions, and study title
- ComprehensiveResponseGenerator's `_get_template_response()` always used the `"explanation"` question type key and returned generic domain templates that didn't address the specific question
- `_make_careless()` returned fully off-topic responses like "ok", "fine", "idk" — real careless participants still write about the topic ("trump is ok i guess")
- Basic text generator used consumer/product language defaults (`"item"`, `"aspect"`) producing "This item is good, the aspect is nice" for ALL questions
- Condition modifiers (e.g., "AI-recommended") were applied universally, even to political studies

**Fixes applied (v1.0.3.8):**
- **Preview**: Complete rewrite of `_get_sample_text_response()` — now accepts `question_text`, `question_context`, `condition`, `study_title` and extracts key themes/subjects to generate topical responses
- **ComprehensiveResponseGenerator**: Added `_extract_response_subject()` and `_generate_context_grounded_response()` — when question context is available, generates responses that directly reference the question topic instead of generic domain templates
- **Careless responses**: Now extract topic words from the already-generated response and build short topic-relevant careless versions
- **Basic text generator**: Context dict now uses question text/context to derive meaningful `topic`, `stimulus`, `product`, `feature` values instead of defaults
- **Condition modifiers**: Only applied for relevant domains (consumer, AI, advertising) — not for political, health, etc.
- **Empty fallbacks**: All fallback responses now include topic words instead of generic meta-commentary

**Further hardening (v1.0.3.9):**
- **`_personalize_for_question()` in response_library.py**: Removed consumer-specific topic intros ("About the product", "For this purchase") that were injected into ALL domains. Replaced with domain-neutral intros ("When making this decision,", "I believe that"). Condition extensions (AI, hedonic, utilitarian) now gated by domain detection — only applied to consumer/AI/advertising domains.
- **`_extend()` in response_library.py**: Expanded domain-specific extensions from 5 domains to 11 (added POLITICAL, POLARIZATION, INTERGROUP, IDENTITY, NORMS, TRUST). General fallback extensions rewritten to be substantive and domain-neutral.
- **`_apply_deep_variation()` in llm_response_generator.py**: Replaced generic high-verbosity elaborations ("I really feel strongly about this") with neutral continuations ("I could go on about this honestly.", "There's definitely more to say about it.").
- **persona_library.py template overhaul**: ALL `{stimulus}` references replaced with `{topic}` across task_summary templates (engaged, default, satisficer, extreme). Careless templates now topic-aware (`"{topic} idk"` not `"idk"`). Product evaluation and follow-up thought templates made domain-neutral.

**Final hardening (v1.0.3.10):**
- **`_get_template_response()` last-resort fallback**: Changed from `"No specific comment."` to topic-extracting fallback that pulls words from question_text/context. If no topic words found, returns `"I answered based on my honest feelings about this."` instead of generic meta-commentary.
- **`_make_careless()` signature expanded**: Now accepts `question_text` parameter. When topic extraction from the response fails, tries extracting from the original question text before falling back. Ultimate fallback changed from `'it'` to `'the question'` — eliminates `"it is ok i guess"` entirely.
- **Preview very_low templates**: All entries now reference `subject_phrase` (`"trump idk"` not `"idk"`). The `_short_subj` truncated subject is prepended/appended to every very_low template.
- **Preview subject_phrase fallback chain**: `topic_words from question_text` → `topic_words from question_name` → `"the questions asked"` (never `"this topic"`).
- **Empty-response fallback in `generate()`**: Changed `'this'` fallback to extract from question_text directly, ultimate fallback `"what was asked"` instead of bare `"this"`.
- **Silent exception suppression fixed**: `except Exception: pass` in enhanced_simulation_engine.py comprehensive_generator call now logs the error with `logger.debug()` for debugging.
- **`survey_feedback` domain templates**: Completely rewritten from survey meta-commentary ("The survey was well-designed") to sentiment-aligned topic responses ("I shared my honest views and I feel pretty good about the topic overall.").
- **persona_library.py task_summary templates**: Removed all "study", "survey", "questions about" meta-commentary from engaged, satisficer, and default templates. Now first-person topic-focused ("I thought about {topic} carefully" not "The study asked about {topic}").
- **persona_library.py general_feedback templates**: Rewritten from survey feedback ("Good survey overall") to genuine opinion sharing ("I shared my genuine opinions based on my actual experiences").
- **text_generator.py defaults**: `_extract_product()` fallback changed from `"it"` to topic-extraction from context; `_extract_product_type()` from `"product"` to `"option"`; `_extract_action()` from `"purchase"` to `"engage"`; `_random_feature()` replaced consumer terms with neutral ones ("approach", "content", "structure").
- **Markov chain extension fallback**: Neutral additions changed from consumer language ("It was okay.", "Average experience overall.") to neutral first-person ("I don't feel too strongly either way.").

**Key principle: NO response should EVER be off-topic.**
- High quality: Detailed, specific, directly addresses the question topic
- Medium quality: Brief but still about the topic
- Low quality: Very short but topic-relevant ("trump is ok i guess")
- Very low quality: May be gibberish but topic words when possible
- Careless: Short and lazy, but STILL about the actual topic

**Comprehensive fallback chain (every level must extract topic):**
1. **First choice**: Extract topic from `question_context` (user-provided context on Design page)
2. **Second choice**: Extract topic words from `question_text` (the actual question asked)
3. **Third choice**: Extract from `question_name` / variable name (e.g., "trump_feelings" → "trump feelings")
4. **Fourth choice**: Use `study_domain` (e.g., "political psychology" → "political psychology")
5. **Last resort**: Use `"the questions asked"` or `"what was asked"` — NEVER `"this topic"`, `"it"`, `"this"`, or bare pronouns

**Topic extraction stop-word pattern (used consistently across all files):**
```python
_stop = {'this', 'that', 'about', 'what', 'your', 'please', 'describe',
         'explain', 'question', 'context', 'study', 'topic', 'condition',
         'think', 'feel', 'have', 'some', 'with', 'from', 'very', 'really'}
```
Always filter stop words from extracted topic, take first 2-4 content words, join into phrase.

**Context flow through the pipeline:**
1. User provides `question_context` via OE context input on Design page
2. Engine embeds context into `question_text`: `"Question: ...\nContext: ...\nStudy topic: ...\nCondition: ..."`
3. LLM generator: Extracts context block, builds prominent `╔══ QUESTION CONTEXT ══╗` section in prompt
4. ComprehensiveResponseGenerator: Extracts embedded context, calls `_generate_context_grounded_response()`
5. Basic text generator (persona_library.py): Uses context dict with topic-enriched `{topic}`, `{product}`, `{feature}` placeholders

**Iterative improvement methodology (v1.0.3.8 → v1.0.3.9 → v1.0.3.10):**
- v1.0.3.8: Fixed the architecture (preview rewrite, context grounding, careless on-topic, condition gating)
- v1.0.3.9: Fixed the content layer (domain-specific extensions, template language, consumer bias removal)
- v1.0.3.10: Fixed the edge cases (every fallback path, silent exceptions, meta-commentary templates, pronoun fallbacks)
- **Lesson**: Generic responses hide in fallback paths, not in primary code. The primary path may be perfect while 10+ fallback paths produce "it is ok i guess" or "No specific comment." Always audit EVERY fallback, EVERY default value, EVERY empty-string handler.

**Provider chain (Gemini already #1):**
- Google AI Studio Gemini 2.5 Flash Lite (10 RPM, 250K TPM, 20 RPD)
- Google AI Studio Gemma 3 27B (30 RPM, 15K TPM, 14,400 RPD)
- Groq Llama 3.3 70B (~30 RPM, 14,400 RPD)
- Cerebras Llama 3.3 70B (~30 RPM, 1M tokens/day)
- OpenRouter Mistral Small 3.1 (varies)

### Documentation Philosophy

- **Update docs with code**: Never leave documentation stale
- **Changelog is essential**: Track what changed and why
- **Version in multiple places**: README, docstrings, __version__ variables
- **Include session links**: Traceability helps debug later issues

---

## Business Brainstorm

### 5 Enterprise/Business Improvement Ideas

#### 1. User Accounts + Persistent Project Workspaces (PREREQUISITE FOR ALL OTHERS)
- **Problem:** Everything is anonymous and session-only. Close browser = lose work. No login, no history, no saved designs.
- **Solution:** Auth (email/password + Google/Microsoft OAuth + SAML for enterprise) + persistent workspace (save, load, duplicate, organize simulation projects).
- **Why it matters:** Foundation for billing, analytics, collaboration. Solves #1 UX pain point (losing work on refresh). Required by institutional procurement. Enables usage tracking and tiered access.
- **Key components:**
  - Auth layer (Streamlit emerging auth patterns, or wrap with FastAPI)
  - Project save/load (serialize session state → database)
  - Project browser (list, search, clone, delete)
  - Role-based access (Student, Instructor, Admin)

#### 2. Tiered Pricing with Stripe Integration
- **Problem:** App is 100% free with no limits. LLM API calls cost money but generate zero revenue.
- **Tiers:**
  - **Free:** 5 sims/mo, max N=200, basic output, no API
  - **Academic ($29/mo):** Unlimited sims, max N=5,000, full output, email support
  - **Professional ($99/mo):** Max N=10,000, full + white-label output, API (100 calls/day), priority support
  - **Enterprise (custom):** Unlimited everything, SSO/SAML, dedicated support
- **Why it matters:** Sustainable revenue, LLM costs are real and growing, academic pricing familiar to universities, enterprise tier opens institutional deals ($10k-$50k/year).

#### 3. LMS Integration (Canvas, Moodle, Blackboard)
- **Problem:** Instructors are core audience (app has instructor reports, group management, difficulty levels) but must manually coordinate outside their LMS.
- **Solution:** LTI (Learning Tools Interoperability) plugin — tool embeds inside Canvas/Moodle/Blackboard. Instructors create "simulation assignments," students complete in-LMS, results flow to gradebook.
- **Why it matters:** Massive distribution channel (every university uses LMS), removes adoption friction, high switching costs, enables auto student tracking. Existing group management + instructor report infrastructure is a head start.
- **Key components:**
  - LTI 1.3 provider implementation
  - Assignment template system (instructor pre-configures, students generate)
  - Grade passback (simulation quality score → LMS gradebook)
  - Roster sync (auto-create student accounts from LMS)

#### 4. REST API + SDK for Programmatic Access
- **Problem:** Power users (computational researchers, method developers, CI/CD pipelines) can't automate simulations — everything requires clicking through Streamlit UI.
- **Solution:** REST API (FastAPI wrapper around existing simulation engine) + Python SDK (`pip install simtool`) + R package.
- **Example:**
  ```python
  from simtool import SimulationClient
  client = SimulationClient(api_key="sk-...")
  result = client.simulate(
      qsf_file="study.qsf",
      conditions=["Control", "Treatment"],
      n_per_condition=500,
      difficulty="hard"
  )
  result.to_csv("simulated_data.csv")
  ```
- **Why it matters:** New customer segment (technical users), enables parameter sweeps/Monte Carlo/power simulations, Jupyter/RMarkdown/CI integration, justifies Pro/Enterprise pricing, creates developer ecosystem.
- **Key components:** API key management, rate limiting per tier, batch endpoint, webhook notifications for long jobs.

#### 5. Study Template Marketplace
- **Problem:** Every user starts from scratch. Common paradigms (trust game, ultimatum game, Stroop, IAT, conjoint) get rebuilt repeatedly.
- **Solution:** Curated + community-contributed marketplace of ready-to-use study templates. Browse by domain, click "Use this template," customize, generate.
- **Why it matters:** Dramatically reduces time-to-first-value, content flywheel (contributions attract users), new revenue stream (premium/verified templates), showcases tool range, builds community.
- **Key components:**
  - Template schema (QSF + metadata + recommended settings)
  - Browse/search by domain, paradigm, citation count
  - "Use this template" → pre-fills wizard
  - Community submissions with review workflow
  - Verified badges, citation tracking

### Implementation Roadmap

**Phase 1 — Foundation (Months 1-2):**
- Weeks 1-2: Auth system (email/password + Google OAuth)
- Weeks 3-4: Project persistence (save/load/list designs)
- Weeks 5-6: Billing infrastructure (Stripe, tier enforcement)
- Weeks 7-8: Usage tracking (per-user metrics, admin dashboard)

**Phase 2 — Revenue (Months 3-4):**
- Weeks 9-10: Launch paid tiers (Free/Academic/Professional)
- Weeks 11-12: API v1 (core simulation endpoints + Python SDK)
- Weeks 13-14: Template marketplace v1 (20 curated templates)
- Weeks 15-16: Team workspaces (shared projects, basic collaboration)

**Phase 3 — Scale (Months 5-6):**
- Weeks 17-18: LMS integration (Canvas LTI plugin beta)
- Weeks 19-20: Enterprise features (SAML SSO, admin dashboard, white-label)
- Weeks 21-22: R SDK + API v2 (batch endpoints, webhooks)
- Weeks 23-24: Community templates (submission workflow, verification)

### Revenue Projection
- **Year 1:** ~$90K (100 Academic + 20 Professional + 3 Enterprise)
- **Year 2:** ~$440K (with LMS channel driving adoption)

### Strategic Priority
**Start with Idea #1 (User Accounts + Workspaces)** — it's the prerequisite for everything else. Without it, you can't bill, can't track, can't collaborate, can't do enterprise sales. It's the keystone.

---

## Simulation Realism Improvement Plan (v1.0.4.3 → v1.0.5.x)

### Audit Summary (2026-02-11)

Comprehensive audit of the simulation pipeline identified 12 improvement areas across 5 files. The engine is strong on political/economic game realism but has systematic gaps in other domains. Key findings:

1. **STEP 2 domain keyword→effect mappings**: 16 domains implemented but several are thin (embodiment: 3 rules, time/temporal: 2 rules). Missing domains: stereotype threat, sunk cost escalation, linguistic styles, deception detection.
2. **Reverse item × acquiescence interaction**: `is_reverse` parameter exists but only flips the scale. Should interact with acquiescence (acquiescent respondents incorrectly agree with reverse-coded items at higher rates) and engagement (careless responders fail reverse items).
3. **Study context underutilized in trait modifiers**: `_get_condition_trait_modifier()` only reads condition text. Study title/description contain domain signals that should enrich trait modification (e.g., "dictator game" in study title + "high trustworthiness" condition = double effect).
4. **Missing domain-specific personas**: No dedicated personas for clinical/anxiety, legal/forensic, sports/competition, relationships/attachment, financial decision-making, or communication/media.
5. **Social desirability bias is domain-agnostic**: SD effect (STEP 9) applies equally to all DVs. In reality, SD effects are strongest for sensitive topics (prejudice, aggression, substance use) and weakest for behavioral/factual reports.
6. **Response library thin domains**: Several domains have <15 template sentences vs. 100+ for rich domains. Survey_feedback domain still contains meta-commentary.
7. **Scale type detection limited**: Only 4 categories (slider, Likert, bipolar, WTP). Missing: matrix scales, forced choice, semantic differential.

### Implementation Plan: 3 Iterations

#### Iteration 1: Simulation Engine Core — Reverse Items, Study Context, STEP 2 Domains
**Files**: `enhanced_simulation_engine.py`

1. **Enhance reverse-item modeling (STEP 5)**
   - Current: Simple flip `center = scale_max - (center - scale_min)` + acquiescence noise
   - New: Careless respondents fail to reverse at rate proportional to `(1 - attention_level)`
   - New: Engagement-dependent reversal accuracy — engaged respondents reverse correctly, satisficers often ignore reversal
   - Scientific basis: Woods (2006): 10-15% of respondents ignore item directionality; Weijters et al. (2010): acquiescence inflates reverse-coded item error by ~0.5 points

2. **Integrate study context into trait modifiers**
   - Pass `self.study_title` and `self.study_description` to `_get_condition_trait_modifier()`
   - Detect study-level domain (health, political, consumer, etc.) from title/description
   - Apply domain-based trait adjustments that interact with condition-level modifiers
   - Example: "Political identity" study + "control" condition → still add political extremity (+0.08) because even control groups in political studies show domain priming

3. **Expand STEP 2 with thin/missing domains**
   - Add: Deception/dishonesty (honest vs. dishonest conditions, die-rolling paradigms)
   - Add: Stereotype threat (diagnostic vs. non-diagnostic test framing)
   - Add: Sunk cost (invested vs. not invested in prior commitment)
   - Add: Construal level (abstract/why vs. concrete/how)
   - Add: Reactance (freedom threat vs. choice conditions)
   - Expand: Embodiment (facial feedback, power pose, touch, warmth/cold priming)
   - Expand: Temporal (time scarcity, deadline, temporal reframing, future time perspective)
   - Each with published effect sizes and references

#### Iteration 2: Persona Library Expansion + Domain-Condition Interactions
**Files**: `persona_library.py`, `enhanced_simulation_engine.py`

1. **Add 6+ new domain-specific personas**
   - Clinical/Anxiety Persona: High anxiety, avoidant, low confidence (Clark & Watson, 1991)
   - Legal/Forensic Persona: Authority-sensitive, justice-focused, literal (Tyler, 2006)
   - Sports/Competition Persona: High achievement motivation, competitive, risk-taking (Vealey, 1986)
   - Relationships/Attachment Persona: Attachment anxiety/avoidance, intimacy needs (Brennan et al., 1998)
   - Financial Decision Persona: Loss-averse, overconfident, status-quo bias (Barber & Odean, 2001)
   - Media/Communication Persona: Source-critical, information-seeking, persuasion knowledge (Friestad & Wright, 1994)

2. **Add domain-condition interaction patterns to STEP 3**
   - Power/hierarchy conditions: boost authority-sensitivity traits
   - Competition conditions: boost achievement motivation, reduce cooperation
   - Mindfulness/meditation conditions: boost attention, reduce extremity
   - Accountability conditions: boost accuracy motivation, reduce SD bias
   - Goal-setting conditions: boost engagement, consistency
   - Priming conditions: detect prime type (semantic, identity, mortality) and adjust relevant traits

#### Iteration 3: Social Desirability Domain Sensitivity + Documentation
**Files**: `enhanced_simulation_engine.py`, `response_library.py`, `docs/CHANGELOG.md`

1. **Make social desirability bias domain-sensitive (STEP 9)**
   - Current: Flat `social_des * scale_range * 0.12` for all DVs
   - New: SD multiplier varies by construct sensitivity
     - Sensitive (prejudice, aggression, substance use, dishonesty): 1.5× SD effect
     - Moderate (prosocial, compliance, health behaviors): 1.0× (default)
     - Low (factual, behavioral frequency, risk perception): 0.5× SD effect
     - Counter (negative self-presentation topics like vulnerability, anxiety): -0.5× (SD inverts)
   - Scientific basis: Nederhof (1985 meta): SD bias d = 0.25-0.75 for sensitive topics; Paulhus (2002): Impression Management vs. Self-Deception differ by domain

2. **Expand _personalize_for_question() domain modifiers**
   - Currently only AI, hedonic, utilitarian get condition-specific modifiers
   - Add: Political condition modifiers ("As someone who leans [direction]...")
   - Add: Health condition modifiers ("Given my health situation...")
   - Add: Moral/ethical modifiers ("From an ethical standpoint...")
   - Add: Intergroup modifiers ("Thinking about the other group...")
   - Add: Financial modifiers ("Considering the financial implications...")

3. **Update documentation**
   - CHANGELOG.md: Full entry for v1.0.4.4 with all changes
   - README.md: Update features section
   - CLAUDE.md: Mark plan items as completed

### v1.0.4.6 Pipeline Quality Overhaul — 10-Step Plan (2026-02-11)

**Problem**: `self.detected_domains` is computed during engine init (5-phase detection from study title, description, conditions) but then **IGNORED by 4 of 5 major methods**. Only persona selection uses it. Effect routing, trait modifiers, calibration, and scaling all independently re-do keyword matching, creating false positives and effect stacking.

#### The 10 Steps

1. **Domain-Aware STEP 2 Routing**: `_get_automatic_condition_effect()` now uses `self.detected_domains` to PRIORITIZE matching domains. Non-detected domains still checked but with 0.5× attenuation to prevent keyword stacking from unrelated domains firing simultaneously.

2. **Domain-Aware Trait Modifiers**: `_get_condition_trait_modifier()` uses `self.detected_domains` directly instead of re-doing keyword matching on study text. Domain-specific trait priming now sourced from the already-detected domain list.

3. **Domain-Aware Persona Weight Adjustment**: `_adjust_persona_weights_for_study()` uses `self.detected_domains` for category boosting instead of re-keyword-matching study text. More precise, eliminates redundant computation.

4. **Domain-Aware Response Calibration**: `_get_domain_response_calibration()` checks `self.detected_domains` as a FIRST pass before falling back to variable-name keyword matching. Enables domain-specific defaults even when variable names are ambiguous.

5. **Persona Domain Mapping Expansion**: Cross-domain persona links added (e.g., `social_comparer` → consumer, `prosocial_individual` → economic_games, `conformist` → organizational). Ensures relevant personas are activated for mixed-domain studies.

6. **Effect Stacking Guard**: Added cumulative effect tracking in STEP 2. When multiple domain keyword sets fire for the same condition, effects are capped to prevent runaway totals from unrelated domains stacking.

7. **Study-Context-Enriched Persona Selection**: Domain detection now ALSO considers `study_context.get("domain")` (user-selected domain from builder/UI), giving it high priority alongside auto-detection.

8. **Domain-Aware Effect Magnitude Scaling (STEP 4)**: `_domain_d_multiplier` now uses `self.detected_domains` as PRIMARY routing instead of keyword re-matching on the full context string. Falls back to keyword matching only for domains not in the detected set.

9. **Persona Pool Validation**: After persona selection, validates that pool has ≥3 domain-specific personas (not just response-style). If too few, broadens to adjacent domains. Logs warning if only response-style personas available.

10. **Cross-DV Coherence Enhancement**: For factorial/multi-condition designs, ensures within-participant response patterns are coherent across DVs (e.g., someone who gives more in dictator game also reports higher trust).

#### Implementation: 3 Iterations
- **Iteration 1** (Steps 1, 2, 3, 6): Domain-aware routing foundation
- **Iteration 2** (Steps 4, 5, 7, 8): Calibration and persona enrichment
- **Iteration 3** (Steps 9, 10): Quality validation and coherence

### v1.0.4.5 Completion Status (2026-02-11)

**COMPLETED** — All 3 iterations implemented:

| Metric | Target | Achieved |
|--------|--------|----------|
| STEP 2 domains with 5+ rules | 18 | 18 (expanded 7 thin domains) |
| Condition categories in STEP 3 | 35+ | 35+ (no change, already met) |
| Persona × condition interactions | 10+ | 13 (added 5 new) |
| Domain personas | 52+ | 52+ (added 6 new) |
| Reverse-item engagement failure | Yes | Yes (3-tier: careless/satisficer/engaged) |
| SD domain sensitivity | Yes | Yes (5 categories + economic game override) |
| SD × reverse interaction | Yes | Yes (attenuate/amplify by reversal status) |

### Next Improvement Targets (v1.0.5.x)

**Remaining from audit that were NOT addressed in v1.0.4.5:**

1. **Response library question-type diversity**: All 105 domains only have `"explanation"` type. Adding justification, evaluation, assessment types would improve specificity.
2. **Narrative transportation domain**: Green & Brock (2000) — emotional narrative involvement predicts persuasion. Not implemented.
3. **Construct accessibility & recency**: Higgins et al. (1977) — recently activated constructs more accessible. Would require tracking prior question sequence.
4. **Scale type detection expansion**: Only 4 categories (slider, Likert, bipolar, WTP). Missing: matrix scales, forced choice, semantic differential.
5. **LLM response validation layer**: Currently only checks `len(r.strip()) >= 3`. Could add off-topic detection, generic meta-commentary screening.
6. **Authority/NFC persona-level interaction in STEP 3** trait modifiers (currently only in STEP 4a interaction effects).
7. **Careless responder cross-item tracking**: Track whether reverse-item failure is consistent across items for same participant.

### Scientific References for Planned Additions

| Improvement | Key Reference | Expected Effect |
|-------------|--------------|-----------------|
| Reverse item failure | Woods (2006) | 10-15% ignore directionality |
| Acquiescence × reverse | Weijters et al. (2010) | +0.5 point error inflation |
| Stereotype threat | Nguyen & Ryan (2008 meta) | d = 0.26 |
| Sunk cost | Arkes & Blumer (1985) | d = 0.30-0.50 |
| Construal level | Trope & Liberman (2010) | Modulates evaluation abstraction |
| Reactance | Brehm & Brehm (1981) | d = 0.30-0.40 |
| SD domain sensitivity | Nederhof (1985 meta) | d = 0.25-0.75 for sensitive topics |
| Power pose effects | Cuddy et al. (2018 reanalysis) | d = 0.10-0.25 (small) |
| Clinical anxiety traits | Clark & Watson (1991) | Tripartite model calibrations |
| Attachment dimensions | Brennan et al. (1998) | ECR anxiety/avoidance norms |
