"""The Design page's "Qs" count and DV "Type" are one fact, not two (v1.3.1.3).

A tester reported that a DV shown as "Single question (always 1)" still accepted 6 in the
"Qs" box and nothing stopped them. The count was accepted, but the keyed Type selectbox kept
its stored value, so the count-driven upgrade (v1.1.1.3) was undone on the same rerun and
``confirmed_scales`` ended up holding ``type='single_item'`` with ``num_items=6`` -- a spec
that generates six columns while the type says one question, and that rides the numeric-DV
path in the engine (``_NUMERIC_DV_TYPES``).

The rule these tests pin down: whichever of the two the user edited last wins.
  * Raise the count on a single-item DV -> it becomes numbered items (count kept).
  * Pick "Single Item" in the Type dropdown -> the count is forced back to 1.
  * A multi-item battery is untouched by either rule.
  * ``_normalize_scale_specs`` reconciles the pair for any spec reaching the engine.
"""
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_APP_DIR = _TESTS_DIR.parent / "simulation_app"
for _p in (str(_APP_DIR), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the synthetic-QSF builders and the AppTest helpers of the design-wiring suite.
# The ``app_test`` fixture is declared below rather than imported: importing a fixture and then
# naming it in every test signature is a redefinition ruff rejects (F811).
from test_design_wiring_v1291 import (  # noqa: E402
    _confirm,
    _dv_rows,
    _no_exception,
    _open_design,
    _sq,
    _start,
    _upload,
    essay,
    likert_matrix,
    make_qsf,
)


@pytest.fixture()
def app_test(monkeypatch, tmp_path):
    """A fresh AppTest with its data/ folders in a temp dir (same contract as the wiring suite)."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    usage_counter = _APP_DIR / ".usage_counter.json"   # written next to app.py after a generation
    existed = usage_counter.exists()
    yield AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    if not existed and usage_counter.exists():
        usage_counter.unlink()


def single_choice(qid, tag, text, points=7):
    """One standalone rating question -- what the parser calls a 'single_item' DV."""
    return _sq(qid, tag, text, "MC", "SAVR",
               Choices={str(i): {"Display": str(i)} for i in range(1, points + 1)},
               ChoiceOrder=list(range(1, points + 1)))


def survey_single_and_battery():
    return make_qsf("Single item and battery", [
        single_choice("QID1", "Fair", "How fair was the offer?"),
        likert_matrix("QID2", "Trust", ["I trust it", "It is reliable", "It is honest"]),
        essay("QID3", "Why", "Please explain your choice"),
    ])


def _row(at, name):
    rows = _dv_rows(at)
    index = [r[0] for r in rows].index(name)
    return index, rows[index]


def _scale(at, name):
    return [s for s in at.session_state["confirmed_scales"] if s["variable_name"] == name][0]


# ---------------------------------------------------------------------------------------------------------------------
# The reported defect: "Qs" = 6 on a single-question DV
# ---------------------------------------------------------------------------------------------------------------------
def test_raising_the_question_count_converts_the_dv_instead_of_leaving_a_contradiction(app_test):
    at = app_test
    _start(at)
    _upload(at, "single_and_battery.qsf", survey_single_and_battery())
    _open_design(at)
    index, row = _row(at, "Fair")
    assert row[1] == "single_item" and row[2] == 1

    at.number_input(key=f"dv_items_v0_{index}").set_value(6)
    at.run()
    _no_exception(at)

    index, row = _row(at, "Fair")
    assert row[2] == 6, "the count the user typed must be kept"
    assert row[1] == "numbered_items", "the Type dropdown must follow the count, not override it"
    scale = _scale(at, "Fair")
    assert (scale["type"], scale["num_items"], scale["items"]) == ("numbered_items", 6, 6)

    # and the spec that reaches the engine carries the same pair
    _confirm(at)
    final = [s for s in at.session_state["inferred_design"]["scales"] if s["variable_name"] == "Fair"][0]
    assert (final["type"], final["num_items"]) == ("numbered_items", 6)


def test_choosing_single_item_forces_the_question_count_back_to_one(app_test):
    at = app_test
    _start(at)
    _upload(at, "single_and_battery.qsf", survey_single_and_battery())
    _open_design(at)
    index, _ = _row(at, "Trust")
    assert _row(at, "Trust")[1][2] == 3                     # arrived as a 3-item battery

    at.selectbox(key=f"dv_type_v0_{index}").set_value("single_item")
    at.run()
    _no_exception(at)

    row = _row(at, "Trust")[1]
    assert (row[1], row[2]) == ("single_item", 1), "one question means a count of one"
    scale = _scale(at, "Trust")
    assert (scale["type"], scale["num_items"]) == ("single_item", 1)

    # raising the count again converts it back rather than reverting the count
    at.number_input(key=f"dv_items_v0_{index}").set_value(4)
    at.run()
    _no_exception(at)
    row = _row(at, "Trust")[1]
    assert (row[1], row[2]) == ("numbered_items", 4)


def test_a_multi_item_battery_is_untouched_by_the_reconciliation(app_test):
    at = app_test
    _start(at)
    _upload(at, "single_and_battery.qsf", survey_single_and_battery())
    _open_design(at)
    index, row = _row(at, "Trust")
    assert (row[1], row[2]) == ("matrix", 3)

    at.number_input(key=f"dv_items_v0_{index}").set_value(5)
    at.run()
    _no_exception(at)
    row = _row(at, "Trust")[1]
    assert (row[1], row[2]) == ("matrix", 5), "a battery keeps its type when the count changes"
    assert _scale(at, "Trust")["num_items"] == 5


# ---------------------------------------------------------------------------------------------------------------------
# The generation path: no contradictory spec can reach the engine
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_items,expected_type", [(1, "single_item"), (2, "numbered_items"), (6, "numbered_items")])
def test_the_normalizer_never_hands_the_engine_a_single_item_with_several_questions(num_items, expected_type):
    import app as appmod

    out = appmod._normalize_scale_specs([
        {"name": "Fair", "variable_name": "Fair", "type": "single_item",
         "num_items": num_items, "scale_points": 7, "scale_min": 1, "scale_max": 7},
    ])
    assert len(out) == 1
    assert (out[0]["type"], out[0]["num_items"]) == (expected_type, num_items)


def test_the_normalizer_leaves_every_other_type_alone():
    import app as appmod

    specs = [
        {"name": "Trust", "variable_name": "Trust", "type": "matrix", "num_items": 3, "scale_points": 7},
        {"name": "Budget", "variable_name": "Budget", "type": "slider", "num_items": 1, "scale_points": 151},
        {"name": "Income", "variable_name": "Income", "type": "numeric_input", "num_items": 1, "scale_points": 100},
        {"name": "Items", "variable_name": "Items", "type": "numbered_items", "num_items": 4, "scale_points": 5},
    ]
    out = appmod._normalize_scale_specs(specs)
    assert [(s["type"], s["num_items"]) for s in out] == [
        ("matrix", 3), ("slider", 1), ("numeric_input", 1), ("numbered_items", 4),
    ]
