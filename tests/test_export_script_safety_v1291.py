"""The exported analysis scripts are run by whoever receives the ZIP: user-typed names must stay data."""
import ast
import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.enhanced_simulation_engine import EnhancedSimulationEngine, _script_name_ok, _script_text  # noqa: E402

MARKER = "PWN_MARKER"
QUOTE_PAYLOAD = "'] ; __import__('os').system('" + MARKER + "') ; x=['"
COMMENT_PREFIXES = ("#", "*", "//")


def _engine(title="Study", conditions=("Control", "Treatment"), scale_name="Trust", seed=3):
    scales = [{"name": scale_name, "variable_name": scale_name, "num_items": 3, "scale_points": 7, "scale_min": 1, "scale_max": 7},
              {"name": "Plain", "variable_name": "Plain", "num_items": 2, "scale_points": 7, "scale_min": 1, "scale_max": 7,
               "reverse_items": [2]}]
    eng = EnhancedSimulationEngine(study_title=title, study_description="d", sample_size=30, conditions=list(conditions),
                                   factors=[{"name": "F", "levels": list(conditions)}], scales=scales, additional_vars=[],
                                   demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, open_ended_questions=[],
                                   seed=seed, allow_template_fallback=True)
    eng.llm_generator.disable_permanently("test")
    df, _meta = eng.generate()
    return eng, df


def _scripts(eng, df):
    return {"python": eng.generate_python_export(df), "r": eng.generate_r_export(df), "julia": eng.generate_julia_export(df),
            "spss": eng.generate_spss_export(df), "stata": eng.generate_stata_export(df)}


ALLOWED_PYTHON_CALLS = {"read_csv", "Categorical", "exists", "merge", "copy", "print", "len", "mean", "groupby", "isna"}


def _called_names(script: str) -> set:
    """Names of everything the Python script calls (so the check ignores marker text inside string literals)."""
    names = set()
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", "?"))
    return names


def _only_in_comments_or_data(script: str) -> bool:
    """Every line carrying the marker is a comment, or has only identifier-safe text around it (no quotes/brackets broke out)."""
    for line in script.splitlines():
        if MARKER in line and not line.lstrip().startswith(COMMENT_PREFIXES):
            if "__import__" in line or ".system(" in line.replace("os_system", ""):
                return False
    return True


@pytest.fixture(scope="module")
def hostile_scale():
    return _engine(scale_name="Trust\nimport os\nos.system('" + MARKER + "')\n" + QUOTE_PAYLOAD)


@pytest.fixture(scope="module")
def hostile_text():
    title = "Study\nimport os\nos.system('" + MARKER + "') print('" + MARKER + "')"
    return _engine(title=title, conditions=("Control\nos.system('" + MARKER + "')", "Treatment'); os.system('" + MARKER + "'); ('"))


def test_a_hostile_scale_name_never_becomes_code(hostile_scale):
    eng, df = hostile_scale
    scripts = _scripts(eng, df)
    assert _called_names(scripts["python"]) <= ALLOWED_PYTHON_CALLS, _called_names(scripts["python"])
    for language, script in scripts.items():
        assert _only_in_comments_or_data(script), language
        for line in script.splitlines():  # nothing the user typed starts a line of code
            if "os.system" in line or line.startswith("import os\nos"):
                assert line.lstrip().startswith(COMMENT_PREFIXES), (language, line)
    assert "Plain_composite" in scripts["python"] and "Plain_composite" in scripts["r"]


def test_line_breaks_in_titles_and_conditions_cannot_start_new_statements(hostile_text):
    eng, df = hostile_text
    scripts = _scripts(eng, df)
    ast.parse(scripts["python"])  # still valid Python
    tree = ast.parse(scripts["python"])
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and MARKER in ast.dump(n)]
    for language, script in scripts.items():
        for line in script.splitlines():
            if MARKER in line:
                # either a comment line, or the marker sits inside one quoted label on a line that starts with known syntax
                assert line.lstrip().startswith(COMMENT_PREFIXES) or "label define" in line or "condition_" in line or "levels" in line \
                    or "condition_order" in line, (language, line)
        assert " " not in script


def test_helpers_accept_plain_names_and_reject_everything_else():
    for ok in ("Trust_1", "Main_DV_3", "1.9Q_2", "Loyalty.mean"):
        assert _script_name_ok(ok), ok
    for bad in ("a b", "a'b", 'a"b', "a]b", "a;b", "a$b", "a`b", "a\nb", "a(b)", "", "a-b"):
        assert not _script_name_ok(bad), bad
    assert _script_text("one\r\ntwo three\x00four") == "one two three four"


def test_normal_names_are_scripted_exactly_as_before():
    eng, df = _engine(scale_name="Trust")
    scripts = _scripts(eng, df)
    assert "Not scripted" not in "".join(scripts.values())
    assert "data['Trust_composite']" in scripts["python"] and "data$Trust_composite" in scripts["r"]
    assert "COMPUTE Trust_composite" in scripts["spss"] and "egen trust_composite" in scripts["stata"]
