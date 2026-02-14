# Claude Code Development Guidelines

## Key Terminology

- **Persona Pipeline**: domain detection → persona filtering → weight adjustment → assignment → trait generation → response generation. Full chain in `enhanced_simulation_engine.py`.
- **Admin Dashboard**: `?admin=1`, password "Dimant_Admin" (SHA-256 hashed).

---

## ABSOLUTE RULE: Page Layout — Next at Top, Scroll at Bottom

**THESE RULES CANNOT BE OVERRIDDEN UNDER ANY CIRCUMSTANCES.**

- Every wizard page (0, 1, 2, 3) MUST have `↑ Back to top` at the very bottom. NEVER delete it.
- NEVER place "Next"/"Continue" buttons at the bottom. They go at the TOP, under the stepper.
- Stepper (1-2-3-4) is visual-only (not clickable). Navigation via visible "Continue to..." button at top.
- Button only appears when all required fields on the current step are complete.

---

## MANDATORY: PR Link After Every Change

Every response with code changes MUST end with:
```
## PR Link
**https://github.com/eugendimant/research-simulations/pull/new/claude/[branch-name]**
```

---

## ABSOLUTE RULE: Version Synchronization — ALL 9 Locations, EVERY Commit

**A version mismatch causes a VISIBLE ERROR BANNER for all users.**

### The 9 locations (ALL must match):

| # | File | Location |
|---|------|----------|
| 1 | `simulation_app/app.py` | `REQUIRED_UTILS_VERSION = "X.X.X.X"` |
| 2 | `simulation_app/app.py` | `APP_VERSION = "X.X.X.X"` |
| 3 | `simulation_app/app.py` | `BUILD_ID = "YYYYMMDD-vXXXXX-description"` |
| 4 | `simulation_app/utils/__init__.py` | `__version__ = "X.X.X.X"` |
| 5 | `simulation_app/utils/__init__.py` | `Version: X.X.X.X` in docstring |
| 6 | `simulation_app/utils/qsf_preview.py` | `__version__ = "X.X.X.X"` |
| 7 | `simulation_app/utils/response_library.py` | `__version__ = "X.X.X.X"` |
| 8 | `simulation_app/README.md` | `**Version X.X.X.X**` in header |
| 9 | `simulation_app/README.md` | `## Features (vX.X.X.X)` |

Also update: `llm_response_generator.py`, `scientific_knowledge_base.py`, `socsim_adapter.py` `__version__`.

### Workflow BEFORE every commit:

1. **Increment** the LAST digit. If 9, roll to 0 and increment left. Each segment is 0-9 (NEVER `.10`).
2. **Update ALL locations** with the SAME version. The #1 failure is updating one file but not the other.
3. **Update BUILD_ID**: `"YYYYMMDD-vXXXXX-short-description"`
4. **Verify**: `grep -r "OLD_VERSION" simulation_app/ --include="*.py" --include="*.md"` — should return nothing.

The app uses `importlib.reload(utils)` for stale cache recovery. Warning only appears on genuine code mismatch.

---

## Code Quality Standards

### Before Every Commit:
1. `python3 -m py_compile <file>` on ALL modified Python files
2. Verify version numbers synchronized
3. Run: `python3 -m pytest tests/test_e2e.py -v --tb=short`
4. No syntax errors or import failures

### Code Style:
- Type hints, docstrings for public functions, follow existing patterns
- Defensive `get()` for dict access, handle edge cases
- Log errors with `_log()` or `logger.debug()`

---

## Behavioral Data Simulation Requirements

### Trash/Unused Block Handling
- `EXCLUDED_BLOCK_NAMES` (200+ patterns) + `EXCLUDED_BLOCK_PATTERNS` (regex) in qsf_preview.py
- Trash blocks must NEVER appear as conditions

### Condition Detection
- Sources: block names, embedded data, BlockRandomizer
- Never include: trash, admin, structural, or quality control blocks

### DV Detection (`_detect_scales()`)
- 6 types: matrix, numbered items, Likert, sliders, single-item, numeric inputs
- Flag `detected_from_qsf: True`

### Factorial Design
- `_render_factorial_design_table()` for visual setup
- Cross conditions: Factor1_Level × Factor2_Level
- Persist factorial state across navigation

### State Persistence
- `_save_step_state()` before navigation, `_restore_step_state()` at step start
- Persist: conditions, factors, scales, factorial config, sample/effect sizes

### Behavioral Realism
- Effect pipeline: STEP 0 (relational parsing) → STEP 1 (valence) → STEP 2 (domain effects) → STEP 3 (trait modifiers) → STEP 4 (magnitude scaling)
- 'lover'/'hater' are identity markers NOT valence keywords
- Economic game baselines: dictator ~28% (Engel 2011), trust ~50%, ultimatum ~40-50%
- Political + economic game: 1.6× effect multiplier (Dimant 2024)
- Intergroup studies MUST show discrimination effects

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
│   ├── example_files/            # QSF training data
│   └── README.md
├── tests/
│   ├── conftest.py               # Path setup (no sys.path needed in tests)
│   └── test_e2e.py               # Main E2E suite
├── docs/
│   ├── CHANGELOG.md
│   └── DEVELOPMENT_REFERENCE.md  # Historical learnings & completed plans
└── CLAUDE.md                     # This file
```

---

## Commit Message Format

```
vX.X.X.X: Brief description

- Specific change 1
- Specific change 2

https://claude.ai/code/[session-id]
```

---

## Streamlit Key Patterns

- `st.markdown('<div>')` does NOT wrap widgets. Use `st.container()`.
- NEVER use `window.parent.location.href` — destroys session state.
- Navigation: visible `st.button()` at top of each page. No hidden buttons, no JS wiring.
- Execution order fix: use `st.container()` placeholder at top, fill at bottom.
- Builder vs QSF paths have different state keys. Always check which is active.

---

## Anti-Patterns (DO NOT)

1. Big-bang rewrites — use focused iterations
2. Update one version location without the others
3. Assume state persists without explicit save/restore
4. Trust QSF block names without trash filtering
5. Use `st.markdown('<div>')` as wrapper
6. Put "Next" at bottom of page
7. Generate off-topic careless responses ("fine" instead of "trump is ok i guess")
8. Use bare pronouns ('it', 'this') as topic fallbacks
9. Write survey meta-commentary in templates ("the study was interesting")
10. Use `{stimulus}` placeholder — use `{topic}` universally
11. Apply consumer/product modifiers to political/health domains
12. Suppress exceptions silently — always `logger.debug()`
13. Only audit primary code paths — generic responses hide in FALLBACK paths

---

## OE Response Architecture (Current v1.1.0.2)

### Three-level cascade:
1. **LLM Generator** (llm_response_generator.py): Free APIs (Groq → Cerebras → Google AI → OpenRouter → Poe)
2. **ComprehensiveResponseGenerator** (response_library.py): 8 structural archetypes, domain-specific concrete details, natural imperfections, topic naturalization
3. **TextResponseGenerator** (persona_library.py): Basic template fallback

### Key principles:
- NO response should EVER be off-topic
- Every response quality level (high→careless) stays on-topic
- Topic fallback: question_context → question_text → question_name → study_domain → "the questions asked"
- One person, one voice: numeric ratings and OE text must be coherent
- `_build_behavioral_profile()` computes 7-dimensional trait vector per participant
- `_enforce_behavioral_coherence()` validates text-numeric consistency

### Non-LLM response archetypes (v1.0.9.8+):
direct_answer | story_first | reasoning_first | rhetorical | concession | stream | list_style | emotional_burst
- Selected by persona formality × engagement traits
- Concrete domain-specific details (political, economic games, health, consumer, trust, social, moral)
- Natural imperfections: typos, missing apostrophes, comma splices, fragments, missing periods

---

## Deep Reference

For historical improvement plans, completed iterations, scientific references, business brainstorm, and detailed architecture documentation, see: `docs/DEVELOPMENT_REFERENCE.md`
