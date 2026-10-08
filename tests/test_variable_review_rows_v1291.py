"""The variable review table accepts open-ended questions as dicts (the current shape) and as bare names (the old one)."""
import importlib.util
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"


def _load_app():
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    spec = importlib.util.spec_from_file_location("_app_review_rows_test", str(_APP_DIR / "app.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_app_review_rows_test"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


def test_open_ended_questions_as_dicts_or_names_make_rows_without_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = _load_app()
    inferred = {"conditions": ["A", "B"], "scales": [{"name": "Trust", "num_items": 3, "scale_points": 7}],
                "open_ended_questions": [{"variable_name": "Why", "question_text": "Why did you choose that option? " * 5},
                                         {"name": "Comments", "question_text": ""}, "OldStyleName", {}, None]}
    rows = app._build_variable_review_rows(inferred, "", "", None)
    open_ended = {r["Variable"]: r for r in rows if r["Role"] == "Open-ended"}
    assert set(open_ended) == {"Why", "Comments", "OldStyleName"}
    assert open_ended["Why"]["Question Text"].endswith("...") and len(open_ended["Why"]["Question Text"]) <= 63
    assert any(r["Variable"] == "Trust" for r in rows)
