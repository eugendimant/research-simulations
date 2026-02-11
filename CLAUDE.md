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
