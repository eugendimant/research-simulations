# Claude Code Development Guidelines

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

## Testing Changes

Before committing:
1. Run `python3 -m py_compile <file>` on modified Python files
2. Verify version numbers are synchronized
3. Test the app loads without version mismatch warning
4. Verify trash blocks are properly excluded
5. Check that state persists across step navigation

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

### Anti-Patterns to Avoid

1. **Big-bang rewrites**: Break into iterations instead
2. **Forgetting version sync**: Always use the checklist
3. **Assuming state persists**: Explicitly save and restore
4. **Trusting QSF block names**: Always filter for trash
5. **Hardcoding patterns**: Use configurable lists for exclusions
6. **Skipping validation**: Users will find edge cases you missed
7. **Ignoring UI feedback**: Scroll position, visual feedback matter

### Documentation Philosophy

- **Update docs with code**: Never leave documentation stale
- **Changelog is essential**: Track what changed and why
- **Version in multiple places**: README, docstrings, __version__ variables
- **Include session links**: Traceability helps debug later issues
