"""Which published knowledge the engine actually consults (v1.3.0.9, audit item P1-8).

The Stage 1 audit (2026-10-23) found tables and helpers that exist, are documented, and are never
called while data is generated: ``CULTURAL_ADJUSTMENTS`` / ``get_cultural_adjustment``,
``get_meta_analytic_effect``, ``empirical_marginals.marginal_for``, ``corrected_value``,
``honesty_notice``, ``coverage_summary`` and ``audit_table``. Unwired knowledge is not a bug on its
own -- a published table is worth keeping -- but a claim that it shapes the data is, and the two
drift apart silently.

So this module pins the inventory instead of the prose. ``_WIRED`` is consulted during a run;
``_PUBLISHED_ONLY`` is reachable through ``utils`` for a reader, a script or the audit page, and is
NOT consulted. A helper that moves between the two lists fails here, and the fix is to move it
here *and* in the documents that describe it (``docs/EMPIRICAL_PROVENANCE.md``,
``simulation_app/README.md``), never to widen the list quietly.
"""
import ast
import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# consulted while data is generated
_WIRED = {
    "get_construct_norm",        # construct norms -> marginal calibration
    "detect_game_type",          # game rows -> economic-game DV marginals
    "corrected_value",           # registry corrections applied to a game row (v1.3.0.8)
    "adjust_effect",             # literature effects -> replication-adjusted magnitude
    "honesty_notice",            # exported in calibration_provenance (v1.3.0.8)
    "coverage_summary",          # exported in calibration_provenance (v1.3.0.8)
}

# published for a reader, a script or the audit page; NOT consulted during generation
_PUBLISHED_ONLY = {
    "get_cultural_adjustment": "CULTURAL_ADJUSTMENTS is a reference table; no culture is asked for "
                               "or inferred at generation time, and guessing one would be the "
                               "polarity mistake of P0-1 in another coat.",
    "get_meta_analytic_effect": "superseded by literature_effects, which carries the tier and the "
                                "shrinkage; the raw table stays readable for scripts.",
    "marginal_for": "empirical_marginals holds published marginals that no DV route asks for yet.",
    "audit_table": "one row per knowledge-base entry, for the audit page and the tests.",
    "tier_cap_binds": "reports whether the tier weight changes the magnitude (P1-7); diagnostic.",
}

_ENGINE = _APP_DIR / "utils" / "enhanced_simulation_engine.py"


def _called_names(path: Path) -> set:
    """Every name that appears as a CALL in ``path`` (imports and mentions do not count)."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


@pytest.fixture(scope="module")
def engine_calls():
    return _called_names(_ENGINE)


@pytest.mark.parametrize("name", sorted(_WIRED))
def test_knowledge_said_to_shape_the_data_is_actually_called(name, engine_calls):
    assert name in engine_calls, (
        f"{name} is listed as wired but the engine never calls it; either wire it or move it to "
        f"_PUBLISHED_ONLY and correct the documents that claim it is used."
    )


@pytest.mark.parametrize("name", sorted(_PUBLISHED_ONLY))
def test_published_only_knowledge_is_still_published_only(name, engine_calls):
    assert name not in engine_calls, (
        f"{name} is now called during generation. That is welcome, but it has to move to _WIRED "
        f"here and be described in docs/EMPIRICAL_PROVENANCE.md with its provenance label."
    )


@pytest.mark.parametrize("name", sorted(_PUBLISHED_ONLY))
def test_published_only_knowledge_is_reachable_for_a_reader(name):
    import utils
    from utils import empirical_marginals, empirical_registry, scientific_knowledge_base
    assert any(hasattr(mod, name) for mod in
               (utils, scientific_knowledge_base, empirical_registry, empirical_marginals)), name


def test_every_published_only_entry_says_why():
    for name, why in _PUBLISHED_ONLY.items():
        assert len(why) > 40, f"{name} needs a reason a reader can check"
