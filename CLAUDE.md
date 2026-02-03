# Claude Code Development Guidelines

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

## Trash/Unused Block Handling

**CRITICAL: Trash blocks must NEVER affect simulation**

- `EXCLUDED_BLOCK_NAMES` in qsf_preview.py contains 200+ patterns
- `EXCLUDED_BLOCK_PATTERNS` contains regex patterns for detection
- `_is_excluded_block_name()` method checks both
- `_get_condition_candidates()` in app.py filters conditions

Always add new exclusion patterns when discovering new trash block naming conventions.

---

## State Persistence

When users navigate between steps, their selections MUST persist:

- `_save_step_state()` - Call before navigation
- `_restore_step_state()` - Call at step start
- `persist_keys` list defines what to save

---

## DV Detection

The `_detect_scales()` method in qsf_preview.py detects 6 types:
- Matrix scales
- Numbered items (e.g., Scale_1, Scale_2)
- Likert scales
- Sliders
- Single-item DVs
- Numeric inputs

Always include `detected_from_qsf: True` flag to distinguish from manual entries.

---

## Testing Changes

Before committing:
1. Run `python3 -m py_compile <file>` on modified Python files
2. Verify version numbers are synchronized
3. Test the app loads without version mismatch warning
