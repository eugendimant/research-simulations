import importlib.util
import os
import re
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"


def _import_app_module():
    """Import app.py exactly as Streamlit does (exec the module). Returns the
    module on success; raises whatever app.py raises (ImportError, etc.)."""
    _prev = sys.path[:]
    sys.path.insert(0, str(_APP_DIR))
    try:
        spec = importlib.util.spec_from_file_location("_app_smoke", str(_APP_DIR / "app.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # raises SystemExit when it reaches the st runtime
        return m
    finally:
        sys.path[:] = _prev


def test_app_does_not_purge_utils_modules_during_startup():
    """Guard against reintroducing runtime sys.modules purges that caused Streamlit import KeyError."""
    app_text = (_APP_DIR / "app.py").read_text(encoding='utf-8')

    assert 'del sys.modules[_k]' not in app_text
    assert '_stale_keys = [k for k in sys.modules' not in app_text


def test_app_imports_without_error():
    """v1.2.8.2 (PRODUCTION-DOWN GUARD): app.py and its ENTIRE top-level import
    chain must load without ImportError. A single bad/renamed/out-of-sync import
    takes the whole Streamlit app down with a redacted ImportError (the real
    outage this catches). This smoke test fails CI the moment any top-level import
    in app.py — or anything it transitively imports — cannot be resolved."""
    try:
        _import_app_module()
    except SystemExit:
        pass  # app.py reached the Streamlit runtime — all imports already resolved
    except ImportError as e:  # pragma: no cover - this is the failure we guard against
        raise AssertionError(f"app.py failed to import (production-down class): {e}")


def test_app_survives_missing_private_atomic_helper():
    """v1.2.8.2: app.py must NOT hard-crash if group_management lacks the private
    _atomic_write_json helper (the exact production outage: version skew between
    app.py and group_management). The defensive import must fall back locally."""
    sys.path.insert(0, str(_APP_DIR))
    try:
        import utils.group_management as gm
        _saved = getattr(gm, "_atomic_write_json", None)
        if hasattr(gm, "_atomic_write_json"):
            del gm._atomic_write_json
        try:
            try:
                m = _import_app_module()
            except SystemExit:
                m = sys.modules.get("_app_smoke")
            assert m is not None and hasattr(m, "_atomic_write_json"), \
                "app.py did not provide a fallback _atomic_write_json"
            # the fallback must actually write atomically
            import json, tempfile
            _p = os.path.join(tempfile.gettempdir(), "_aw_safety_test.json")
            m._atomic_write_json(_p, {"ok": 1})
            assert json.load(open(_p)) == {"ok": 1}
            os.unlink(_p)
        finally:
            if _saved is not None:
                gm._atomic_write_json = _saved
    finally:
        if str(_APP_DIR) in sys.path:
            sys.path.remove(str(_APP_DIR))


def test_app_has_no_toplevel_private_cross_module_imports():
    """v1.2.8.2: forbid NEW `from x import _private` at app.py top level (module
    scope) unless wrapped in try/except. Importing a private cross-module symbol
    hard at top level is what took the app down — keep it from creeping back."""
    lines = (_APP_DIR / "app.py").read_text(encoding="utf-8").splitlines()
    offenders = []
    for i, line in enumerate(lines, 1):
        # only module-scope (no indentation) bare `from ... import ..._priv`
        if re.match(r"^from\s+[\w.]+\s+import\s+.*\b_[a-z]\w*", line):
            offenders.append(f"{i}: {line.strip()}")
    assert not offenders, (
        "top-level private cross-module import(s) found (wrap in try/except with a "
        f"fallback): {offenders}")
