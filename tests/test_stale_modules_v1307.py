"""A redeploy can leave the previous release's utils modules in memory (v1.3.0.7).

The new app then passes arguments the old engine does not know ("EnhancedSimulationEngine.__init__() got an
unexpected keyword argument 'auto_effects'"). These tests recreate that state in a fresh interpreter and check
that the app's two guards heal it: the module refresh that runs before the imports, and the engine constructor.
"""
import ast
import subprocess
import sys
import textwrap
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"


def _function_source(name: str) -> str:
    source = (APP_DIR / "app.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in app.py")


def _run(code: str) -> str:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                          cwd=str(APP_DIR))
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip().splitlines()[-1]


_STALE_SETUP = textwrap.dedent(f"""
    import sys, inspect, logging
    from typing import Any, List
    sys.path.insert(0, {str(APP_DIR)!r})
    import utils
    import utils.enhanced_simulation_engine as eng
    real_version = utils.__version__

    class OldEngine:  # the previous release: no auto_effects / design arguments
        def __init__(self, study_title, sample_size=10):
            self.study_title, self.sample_size = study_title, sample_size

    # what a redeploy leaves behind: old version string, old engine class bound everywhere
    utils.__version__ = "0.0.0.0"
    eng.EnhancedSimulationEngine = OldEngine
    utils.EnhancedSimulationEngine = OldEngine
""")


def test_stale_utils_modules_are_reloaded_before_the_app_binds_them():
    code = _STALE_SETUP + _function_source("_refresh_stale_utils_modules") + textwrap.dedent("""
        reloaded = _refresh_stale_utils_modules(real_version)
        cls = sys.modules["utils.enhanced_simulation_engine"].EnhancedSimulationEngine
        ok = (utils.__version__ == real_version and "auto_effects" in inspect.signature(cls.__init__).parameters
              and "utils.enhanced_simulation_engine" in reloaded and "utils" in reloaded
              and utils.EnhancedSimulationEngine is cls)
        print("OK" if ok else f"FAIL {utils.__version__} {reloaded[:5]}")
    """)
    assert _run(code) == "OK"


def test_refresh_is_a_no_op_when_the_versions_match():
    code = _STALE_SETUP + _function_source("_refresh_stale_utils_modules") + textwrap.dedent("""
        utils.__version__ = real_version
        print("OK" if _refresh_stale_utils_modules(real_version) == [] else "FAIL")
    """)
    assert _run(code) == "OK"


def test_engine_constructor_heals_an_old_engine_class_instead_of_failing():
    code = _STALE_SETUP + _function_source("_construct_engine") + textwrap.dedent("""
        EnhancedSimulationEngine = OldEngine   # the name app.py bound at import time
        e = _construct_engine(study_title="t", study_description="d", sample_size=20, conditions=["A", "B"], factors=[], scales=[],
                              additional_vars=[], demographics={}, auto_effects=False, seed=1)
        print("OK" if type(e).__name__ == "EnhancedSimulationEngine" and e.auto_effects is False else "FAIL")
    """)
    assert _run(code) == "OK"


def test_engine_constructor_drops_unknown_arguments_as_a_last_resort():
    code = _STALE_SETUP + _function_source("_construct_engine") + textwrap.dedent("""
        EnhancedSimulationEngine = OldEngine
        sys.modules.pop("utils.enhanced_simulation_engine", None)   # nothing to reload
        e = _construct_engine(study_title="t", sample_size=5, auto_effects=False, design=None)
        print("OK" if isinstance(e, OldEngine) and e.sample_size == 5 else "FAIL")
    """)
    assert _run(code) == "OK"
