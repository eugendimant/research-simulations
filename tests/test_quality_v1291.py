"""v1.2.9.1 regression tests: open-ended text quality, numeric text boxes, collector hardening.

Every test here is offline and fast (no network, no LLM calls).
"""

import json
import os
import random
import re
import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"


# ──────────────────────────────────────────────────────────────────────────────
# text_cleanup helpers
# ──────────────────────────────────────────────────────────────────────────────
from utils import text_cleanup as tc  # noqa: E402


def test_trim_dangling_tail_keeps_valid_endings():
    assert tc.trim_dangling_tail("I support her.") == "I support her."
    assert tc.trim_dangling_tail("I care a lot about.") == "I care a lot about."
    assert tc.trim_dangling_tail("the person I voted for.") == "the person I voted for."
    assert tc.trim_dangling_tail("I chose option A.") == "I chose option A."  # label, not an article
    assert tc.trim_dangling_tail("ok") == "ok"


def test_trim_dangling_tail_removes_cut_off_phrases():
    assert tc.trim_dangling_tail("I think the policy is good for the") == "I think the policy is good."
    assert tc.trim_dangling_tail("it was fine but") == "it was fine."
    assert tc.trim_dangling_tail("the person I voted for") == "the person I voted."  # no closing punctuation


def test_lower_first_keeps_proper_nouns_and_acronyms():
    assert tc.lower_first("The policy is fine") == "the policy is fine"
    assert tc.lower_first("I think so") == "I think so"
    assert tc.lower_first("I'm not sure") == "I'm not sure"
    assert tc.lower_first("AI is risky") == "AI is risky"
    assert tc.lower_first("Trump seems fine but I like Trump less") == "Trump seems fine but I like Trump less"


def test_article_agreement_and_labels():
    assert tc.fix_indefinite_articles("It was an cool idea") == "It was a cool idea"
    assert tc.fix_indefinite_articles("a hour ago") == "an hour ago"
    assert tc.fix_indefinite_articles("It is a interesting point.") == "It is an interesting point."
    assert tc.fix_indefinite_articles("a unique view") == "a unique view"
    assert tc.fix_indefinite_articles("Group A and B are fine") == "Group A and B are fine"
    assert tc.fix_indefinite_articles("An AI system") == "An AI system"


def test_contractions_never_turn_its_into_it_has():
    rng = random.Random(1)
    out = tc.apply_contractions("I don't think it's fine and that's why I'm here. It's been a while.",
                                rng, expand=True, prob=1.0)
    assert "it has" not in out.lower().replace("it has been", "")
    assert "It's been" in out  # perfect tense is left alone
    assert "do not" in out and "that is" in out and "I am" in out
    short = tc.apply_contractions("I do not think it is fine and that is why I am here.", rng, expand=False, prob=1.0)
    assert "don't" in short and "it's" in short and "I'm" in short
    # a copula at the end of a clause cannot be contracted ("what it's.")
    assert tc.apply_contractions("that is what it is.", rng, expand=False, prob=1.0) == "that's what it is."


def test_misplaced_filler_removed_but_clause_boundary_filler_kept():
    assert tc.remove_misplaced_fillers("when it honestly, comes to AI") == "when it comes to AI"
    assert tc.remove_misplaced_fillers("it was fine, you know, but I am not sure") == "it was fine, you know, but I am not sure"
    assert "you know, you know" not in tc.remove_misplaced_fillers("i believe you know, you know, influenced")


def test_insert_filler_only_at_clause_boundaries():
    words = "it was fine but I am not sure about the whole thing".split()
    assert tc.insert_filler(words, ", you know,", random.Random(1))
    assert " ".join(words) == "it was fine, you know, but I am not sure about the whole thing"
    plain = "the way things are going".split()
    assert not tc.insert_filler(plain, "I mean", random.Random(1))
    assert plain == "the way things are going".split()


def test_insert_adverb_after_subject():
    w = "I think it is fine".split()
    assert tc.insert_adverb_after_subject(w, "honestly") and " ".join(w) == "I honestly think it is fine"
    w = "The policy is fine".split()
    assert not tc.insert_adverb_after_subject(w, "honestly")


def test_swap_one_word_is_context_checked():
    rng = random.Random(3)
    # negated "really" and "I think about" must not be swapped
    for text in ("I do not really like this", "I think about this a lot"):
        w = text.split()
        assert not tc.swap_one_word(w, rng, formal=False)
        assert " ".join(w) == text
    w = "it is an interesting idea".split()
    assert tc.swap_one_word(w, random.Random(1), formal=False)
    assert tc.fix_indefinite_articles(" ".join(w)) == " ".join(w)  # article already agrees


def test_drop_one_optional_word_never_drops_negation_or_content():
    for seed in range(40):
        w = "I really do not like this and it is just not right".split()
        tc.drop_one_optional_word(w, random.Random(seed))
        joined = " ".join(w)
        assert joined.count("not") == 2 and "like" in joined and "right" in joined


def test_truncate_to_sentences_prefers_sentence_boundaries():
    text = ("I think the policy is fair. The rollout was confusing and nobody explained it well. "
            "I would have liked more time to prepare, but the staff were kind. Overall I am fine with it.")
    assert tc.truncate_to_sentences(text, 8) == "I think the policy is fair."
    assert tc.truncate_to_sentences(text, 20).endswith("explained it well.")
    cut = tc.truncate_to_sentences("One long sentence that keeps going without a stop for the participant to read, "
                                   "and then more words after the comma that tip it over the limit by a", 18)
    assert cut.endswith(".") and not re.search(r"\b(a|the|by)\.$", cut)


def test_english_cleanup_leaves_other_languages_alone():
    spanish = "Voy a España con una amiga y creo que es una idea excelente a la vez"
    italian = "Credo che a una persona come me piaccia a Roma andare a ogni festa"
    for text in (spanish, italian):
        assert not tc.looks_english(text)
        assert tc.finalize_generated_text(text) == text
    assert tc.finalize_generated_text("I think it is a interesting idea and the people agree with it") == \
        "I think it is an interesting idea and the people agree with it"


def test_looks_non_english():
    assert tc.looks_non_english("ta sezione ti chiediamo di rispondere a una domanda")
    assert tc.looks_non_english("¿Cuál es su opinión sobre el tema de la inmigración?")
    assert not tc.looks_non_english("What do you think about the new policy on remote work?")
    assert not tc.looks_non_english("tax cuts for the wealthy")


# ──────────────────────────────────────────────────────────────────────────────
# Question-text cleaning and numeric text boxes (engine helpers)
# ──────────────────────────────────────────────────────────────────────────────
from utils import enhanced_simulation_engine as eng  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("How do you feel about it?&nbsp;<br>Please explain &amp; elaborate.", "How do you feel about it? Please explain & elaborate."),
    ("Please describe the AI&#39;s decision <b>in detail</b>.", "Please describe the AI's decision in detail."),
    ("&amp;quot;Quoted&amp;quot; text", '"Quoted" text'),
    ("Click to write the question text", ""),
    ("Click to write Question Text", ""),
    ("Q104", ""), ("QID5", ""), ("Q4.4_7_TEXT", ""), ("", ""), (None, ""),
    ("Rate ${e://Field/Brand} on a scale", "Rate ${e://Field/Brand} on a scale"),
    ("5 < 7 and 9 > 3", "5 < 7 and 9 > 3"),
    ("Q Learning is a method", "Q Learning is a method"),
])
def test_clean_question_text(raw, expected):
    assert eng._clean_question_text(raw) == expected


@pytest.mark.parametrize("question, extra, kind", [
    ("What is your year of birth?", {}, "year_of_birth"),
    ("How old are you?", {}, "age"),
    ("Age", {}, "age"),
    ("How many tickets would you buy? (enter a number)", {}, "count"),
    ("What percentage of the budget would you allocate?", {}, "percent"),
    ("How much would you pay for this product? ($)", {}, "money"),
    ("What is your zip code?", {"content_type": "ValidZip"}, "zip"),
    ("Enter the number", {"content_type": "ValidNumber", "number_min": 0, "number_max": 100}, "count"),
])
def test_numeric_text_boxes_detected(question, extra, kind):
    spec = eng._infer_numeric_answer_spec(question, "", extra)
    assert spec is not None and spec["kind"] == kind


@pytest.mark.parametrize("question", [
    "Please explain why you chose that amount",
    "How many hours did you sleep and why?",
    "Tell us what you think about the policy",
    "What is your favorite color?",
    "Describe the person in the picture",
    "Click to write the question text",
    "Please enter any other comments",
])
def test_free_text_boxes_stay_free_text(question):
    assert eng._infer_numeric_answer_spec(question, "", {}) is None


@pytest.mark.parametrize("question, kind, pattern", [
    ("Please key in your MTurk ID.&nbsp;", "mturk_id", r"A[A-Z0-9]{12,13}"),
    ("Please enter your worker ID", "mturk_id", r"A[A-Z0-9]{12,13}"),
    ("Welcome. What is your Prolific Academic ID?", "prolific_id", r"[0-9a-f]{24}"),
    ("Enter your participant ID", "participant_id", r"\d{6}"),
    ("Please insert here the code that was given to you when you received this survey:", "participant_id", r"\d{6}"),
    ("Please paste into the field below the survey ID number provided to you on the last page.", "participant_id", r"\d{6}"),
])
def test_identifier_boxes_get_ids_not_prose(question, kind, pattern):
    import numpy as np
    spec = eng._infer_numeric_answer_spec(question, "", {})
    assert spec is not None and spec["kind"] == kind
    rng = np.random.RandomState(2)
    answers = [eng._draw_numeric_answer(spec, rng) for _ in range(50)]
    assert all(re.fullmatch(pattern, a) for a in answers), answers[:3]
    assert len(set(answers)) == len(answers)  # IDs are not shared between participants


@pytest.mark.parametrize("question", [
    "Did you complete this survey on MTurk? Please explain why not.",
    "Please describe your experience on Prolific",
])
def test_id_words_inside_a_real_question_stay_free_text(question):
    assert eng._infer_numeric_answer_spec(question, "", {}) is None


def test_crowd_platform_wording_does_not_turn_other_questions_into_id_boxes():
    assert eng._infer_numeric_answer_spec("What is your age? (MTurk workers only)", "", {})["kind"] == "age"
    count = eng._infer_numeric_answer_spec("How many HITs have you completed on MTurk before? (enter a number)", "", {})
    assert count["kind"] == "count"


def test_age_and_birth_year_boxes_honor_the_declared_validation_range():
    import numpy as np
    rng = np.random.RandomState(4)
    age_spec = eng._infer_numeric_answer_spec("What is your age?", "", {"content_type": "ValidNumber", "number_min": 65, "number_max": 90})
    ages = [int(eng._draw_numeric_answer(age_spec, rng)) for _ in range(400)]
    assert min(ages) >= 65 and max(ages) <= 90
    year_spec = eng._infer_numeric_answer_spec("What is your year of birth?", "", {"content_type": "ValidNumber", "number_min": 1940, "number_max": 1960})
    years = [int(eng._draw_numeric_answer(year_spec, rng)) for _ in range(400)]
    assert min(years) >= 1940 and max(years) <= 1960
    # without a declared range the usual adult distribution applies
    free = [int(eng._draw_numeric_answer(eng._infer_numeric_answer_spec("What is your age?", "", {}), rng)) for _ in range(400)]
    assert 18 <= min(free) and max(free) <= 80


def test_numeric_answers_respect_declared_range_and_are_deterministic():
    import numpy as np
    spec = {"kind": "count", "lo": 0.0, "hi": 10.0, "decimals": False}
    a = [eng._draw_numeric_answer(spec, np.random.RandomState(7)) for _ in range(3)]
    b = [eng._draw_numeric_answer(spec, np.random.RandomState(7)) for _ in range(3)]
    assert a == b
    rng = np.random.RandomState(1)
    vals = [int(eng._draw_numeric_answer(spec, rng)) for _ in range(500)]
    assert min(vals) >= 0 and max(vals) <= 10
    assert 0 in vals and 10 in vals  # people pick the endpoints
    ages = [int(eng._draw_numeric_answer({"kind": "age"}, rng)) for _ in range(500)]
    assert min(ages) >= 18 and max(ages) <= 80
    years = [int(eng._draw_numeric_answer({"kind": "year_of_birth"}, rng)) for _ in range(500)]
    assert 1945 <= min(years) and max(years) <= 2007
    assert re.fullmatch(r"\d{5}", eng._draw_numeric_answer({"kind": "zip"}, rng))


# ──────────────────────────────────────────────────────────────────────────────
# LLM post-processing is grammar-safe
# ──────────────────────────────────────────────────────────────────────────────
from utils.llm_response_generator import LLMResponseGenerator  # noqa: E402

_BASES = [
    "I think the policy is an important step, but I do not trust the people running it. The rollout was very confusing and I really did not understand the rules. Overall I would say it was a difficult experience.",
    "Honestly the AI made a good decision here. It was an interesting choice and I understand why it picked option A over option B. I believe most people would agree because the outcome was positive for everyone.",
    "I do not like how this was handled. It was a bad experience and I am concerned about what comes next. Nobody explained anything to us and that is really not acceptable.",
    "My opinion is that the other participant was fair. I gave them half because it seemed like the right thing to do. I would definitely do the same again, although it was a hard call.",
    "I support the candidate because the economy matters most to me. Trump has been a mixed bag in my view, and I am not sure the alternatives are better. I just want a stable country for my kids.",
    "Group A seemed more cooperative than Group B in my experience. I think that is because they were given more information. It is a good example of why transparency is important.",
]
_NEG = re.compile(r"\b(not|never|no|cannot|nobody|nothing|neither|nor)\b|n't\b", re.I)


def _lint(text, base):
    problems = []
    if re.search(r"\s[,;:!?]", text):
        problems.append("space before punctuation")
    if re.search(r",\s*,", text):
        problems.append("double comma")
    if re.search(r"\b(\w+)\s+\1\b", text, re.I):
        problems.append("repeated word")
    if tc.fix_indefinite_articles(text) != text:
        problems.append("a/an")
    if tc.trim_dangling_tail(text) != text:
        problems.append("cut-off ending")
    if re.search(r"\bto\s+(wanna|gonna|gotta)\b|\b(wanna|gonna|gotta)\s+to\b", text, re.I):
        problems.append("wanna to")
    if re.search(r"\b(of|to|the|a|an|and|but|with|by|for)\s*,", text, re.I) and not re.search(r"used to,", text):
        problems.append("comma after function word")
    tokens = lambda x: set(re.findall(r"[a-z']+", x.lower()))  # noqa: E731
    for sent in re.split(r"(?<=[.!?])\s+", base):
        st = tokens(sent)
        if len(st) >= 5 and len(st & tokens(text)) / len(st) >= 0.7 and len(_NEG.findall(sent)) > len(_NEG.findall(text)):
            problems.append("negation lost")
            break
    return problems


def test_deep_variation_is_grammar_safe_across_personas():
    """500+ persona draws per base text: no mechanical grammar damage and no lost negations."""
    failures = []
    for bi, base in enumerate(_BASES):
        for k in range(120):
            rng = random.Random(bi * 100000 + k)
            pr = random.Random(bi * 7919 + k * 13 + 1)
            profile = None
            if pr.random() > 0.45:
                profile = {"straight_lined": pr.random() < 0.07,
                           "response_pattern": pr.choice(["strongly_positive", "positive", "neutral", "negative", "strongly_negative"]),
                           "intensity": pr.random(),
                           "trait_profile": {"extremity": pr.random(), "social_desirability": pr.random(),
                                             "attention": pr.random(), "consistency": pr.random()}}
            out = LLMResponseGenerator._apply_deep_variation(
                base, pr.random(), pr.random(), pr.uniform(0.05, 1.0), rng, behavioral_profile=profile)
            assert out and len(out.strip()) >= 3
            bad = _lint(out, base)
            if bad:
                failures.append((bad, out))
    assert not failures, failures[:5]


def test_deep_variation_is_deterministic_for_a_seed():
    args = (_BASES[0], 0.6, 0.2, 0.7)
    a = LLMResponseGenerator._apply_deep_variation(*args, random.Random(5))
    b = LLMResponseGenerator._apply_deep_variation(*args, random.Random(5))
    assert a == b


# ──────────────────────────────────────────────────────────────────────────────
# Stylometric engine and validator no longer damage text
# ──────────────────────────────────────────────────────────────────────────────
def test_stylometric_formal_writer_keeps_its_contractions_grammatical():
    from utils.hbs_stylometric_engine import HBSStylometricEngine, StylometricFingerprint
    engine = HBSStylometricEngine()
    fp = StylometricFingerprint(contraction_rate=0.0)  # prefers expanded forms
    out = engine._apply_contractions("it's a great idea and it's clear that I'm right", fp, random.Random(1))
    assert "it has" not in out
    assert "it is a great idea" in out


def test_stylometric_simplification_is_whole_word():
    from utils.hbs_stylometric_engine import HBSStylometricEngine
    engine = HBSStylometricEngine()
    out = engine._simplify_vocabulary("The implementation was demonstrated. We will implement it. Regarding the plan, I purchased one.")
    assert "implementation" in out and "demonstrated" in out and "purchased" in out
    assert "will do it" in out
    assert out.count("About the plan") == 1


def test_validator_uniqueness_uses_natural_tags_not_counters():
    from utils.hbs_validator import HBSValidator
    pd = pytest.importorskip("pandas")
    text = "I think the policy is mostly fair to everyone involved in the process"
    df = pd.DataFrame({"OpenText": [text] * 6, "CONDITION": ["A"] * 3 + ["B"] * 3})
    out = HBSValidator(seed=3)._correct_oe_uniqueness(df.copy())
    vals = [str(v) for v in out["OpenText"]]
    assert len({v.lower() for v in vals}) == len(vals)
    assert not any(re.search(r"\(\d+\)", v) for v in vals)


def test_validator_length_truncation_ends_cleanly():
    from utils.hbs_validator import HBSValidator
    pd = pytest.importorskip("pandas")
    long = " ".join(["The rollout was confusing and nobody explained the rules to the staff in advance."] * 6)
    df = pd.DataFrame({"OpenText": [long] * 5, "CONDITION": ["A"] * 5})
    out = HBSValidator(seed=3)._correct_oe_length(df.copy())
    for v in out["OpenText"]:
        assert str(v).strip().endswith((".", "!", "?"))
        assert not re.search(r"\b(the|a|an|of|to|and|by)\.$", str(v))


def test_validator_does_not_treat_binary_items_as_scale_items():
    from utils.hbs_validator import HBSValidator
    data = {"Q_1": [1, 2, 1, 2, 1, 2, 1, 2], "Q_2": [2, 1, 2, 1, 2, 1, 2, 1], "Q_3": [1, 1, 2, 2, 1, 1, 2, 2]}
    assert HBSValidator()._find_scale_columns(data) == []


# ──────────────────────────────────────────────────────────────────────────────
# QSF collector hardening
# ──────────────────────────────────────────────────────────────────────────────
from utils import github_qsf_collector as coll  # noqa: E402


def test_collector_accepts_every_example_qsf_and_rejects_junk():
    ex = _APP_DIR / "example_files"
    files = sorted(ex.glob("*.qsf"))[:25]
    assert files
    for f in files:
        ok, reason = coll.validate_qsf_payload(f.read_bytes())
        assert ok, f"{f.name}: {reason}"
    for junk in (b"", b"not json", b"[1,2,3]", json.dumps({"a": 1}).encode(), "x".encode() * (coll.MAX_QSF_BYTES + 1)):
        assert not coll.validate_qsf_payload(junk)[0]
    assert not coll.validate_qsf_payload("text")[0]


def test_collector_rate_limit_is_a_sliding_window(monkeypatch):
    coll._upload_times.clear()
    allowed = [coll._allow_upload_now() for _ in range(coll.MAX_UPLOADS_PER_HOUR + 5)]
    assert allowed.count(True) == coll.MAX_UPLOADS_PER_HOUR
    assert allowed[coll.MAX_UPLOADS_PER_HOUR:] == [False] * 5
    # an hour later the window has emptied
    import time as _t
    real = _t.time
    monkeypatch.setattr(coll.time, "time", lambda: real() + 3601)
    assert coll._allow_upload_now()
    coll._upload_times.clear()


def test_collector_filename_is_sanitised_and_capped():
    name = coll._sanitize_filename("../../etc/passwd" + "x" * 400 + ".qsf")
    assert "/" not in name and ".." not in name
    assert len(name) <= coll.MAX_FILENAME_LENGTH + 10
    assert name.endswith(".qsf")


# ──────────────────────────────────────────────────────────────────────────────
# Importing app.py the way the other tests do
# ──────────────────────────────────────────────────────────────────────────────
def _load_app():
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_app_pw_test", str(_APP_DIR / "app.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_app_pw_test"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


# ──────────────────────────────────────────────────────────────────────────────
# Exported analysis scripts (R / Python / Julia / SPSS / Stata)
# ──────────────────────────────────────────────────────────────────────────────
def _script_engine():
    eng_cls = eng.EnhancedSimulationEngine
    e = eng_cls(
        study_title="Script check", study_description="d", sample_size=20, conditions=["A", "B"], factors=[],
        scales=[
            {"name": "Mood", "variable_name": "Mood", "num_items": 3, "scale_points": 11,
             "scale_min": 0, "scale_max": 10, "reverse_items": [2]},
            {"name": "Trust", "variable_name": "Trust", "num_items": 3, "scale_points": 7, "reverse_items": [3]},
        ],
        additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[], seed=1)
    e.llm_generator.disable_permanently("test")
    df, _ = e.generate()
    return e, df


def test_exported_scripts_read_the_file_that_is_in_the_zip():
    e, df = _script_engine()
    for lang in ("r", "python", "julia", "spss", "stata"):
        text = getattr(e, f"generate_{lang}_export")(df)
        assert "Simulated_Data.csv" in text, lang
        assert not re.search(r"Simulated\.csv", text), lang


def test_exported_scripts_reverse_code_with_min_plus_max_and_use_it_in_composites():
    e, df = _script_engine()
    r = e.generate_r_export(df)
    assert "data$Mood_2_R <- 10 - data$Mood_2" in r          # 0-10 scale: 10 - x
    assert "data$Trust_3_R <- 8 - data$Trust_3" in r          # 1-7 scale: 8 - x
    assert "rowMeans(cbind(data$Mood_1, data$Mood_2_R, data$Mood_3)" in r
    py = e.generate_python_export(df)
    assert "data[['Mood_1', 'Mood_2_R', 'Mood_3']].mean(axis=1)" in py
    assert "MEAN(Mood_1 Mood_2_R Mood_3)" in e.generate_spss_export(df)
    assert "rowmean(mood_1 mood_2_r mood_3)" in e.generate_stata_export(df)
    assert ":Mood_2_R" in e.generate_julia_export(df)


# ──────────────────────────────────────────────────────────────────────────────
# Condition detection ignores Qualtrics piped text; secrets never raise
# ──────────────────────────────────────────────────────────────────────────────
def test_piped_text_and_formulas_are_not_conditions():
    from utils.qsf_preview import QSFPreviewParser
    kept = QSFPreviewParser()._dedupe_conditions([
        "Collective Condition", "{e://Field/participantId}", "${e://Field/projectId}",
        "$e{ q://QID81/ChoiceTextEntryValue/1 + q://QID81/ChoiceTextEntryValue/2 }", "Individuated Condition",
    ])
    assert kept == ["Collective Condition", "Individuated Condition"]


def test_missing_secrets_file_does_not_break_a_run():
    app = _load_app()
    assert app._secret("DEFINITELY_NOT_SET", "fallback") == "fallback"


def test_preview_answers_numeric_text_boxes_with_numbers():
    app = _load_app()
    df = app._generate_preview_data(
        conditions=["A", "B"],
        scales=[{"name": "Trust", "variable_name": "Trust", "num_items": 3, "scale_points": 7}],
        open_ended=[{"variable_name": "age_q", "question_text": "What is your age?&nbsp;"},
                    {"variable_name": "tickets", "question_text": "How many tickets will you get? (enter a number)"},
                    {"variable_name": "why", "question_text": "Please explain your answer"}],
        n_rows=5, difficulty="medium", study_title="Study", study_description="A study of trust")
    for col in ("age_q", "tickets"):
        shown = [str(v) for v in df[col].head(3)]
        assert all(re.fullmatch(r"\d+", v) for v in shown), (col, shown)
    assert not re.fullmatch(r"\d+", str(df["why"].iloc[0]))


def test_text_boxes_that_duplicate_numeric_dvs_are_skipped():
    oe = [{"name": "Score_Total", "variable_name": "Score_Total"}, {"name": "why", "variable_name": "why"},
          {"name": "Trust", "variable_name": "Trust"}]
    scales = [{"name": "Score Total", "variable_name": "Score_Total", "type": "numeric_input"},
              {"name": "Trust", "variable_name": "Trust", "type": "matrix"}]
    kept, dropped = eng._drop_oe_duplicating_dvs(oe, scales)
    assert dropped == ["Score_Total"]                      # a numeric DV already has its numbers
    assert [q["name"] for q in kept] == ["why", "Trust"]   # a Likert matrix of the same name is left alone


def test_collector_duplicate_check_queries_the_configured_branch(monkeypatch):
    """With GITHUB_QSF_BRANCH set, "does this file exist" must look at that branch, not the default."""
    import types

    seen = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        seen["params"] = params
        return types.SimpleNamespace(status_code=200, json=lambda: [{"name": "other.qsf"}])

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=fake_get))
    config = {"token": "t", "repo": "o/r", "path": "p", "branch": "qsf-collection"}
    assert coll._file_exists_in_repo("new.qsf", config) is False
    assert seen["params"] == {"ref": "qsf-collection"}
    config.pop("branch")
    coll._file_exists_in_repo("new.qsf", config)
    assert seen["params"] == {"ref": "main"}


def test_email_send_still_works_with_configured_secrets(monkeypatch):
    """The SMTP path reads its settings through _secret(); with secrets present it must behave as before."""
    import smtplib

    import streamlit as st

    app = _load_app()
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None, **_kw):
            sent["host"], sent["port"] = host, port

        def ehlo(self):
            pass

        def starttls(self, context=None):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg):
            sent["to"] = msg["To"]
            sent["subject"] = msg["Subject"]
            sent["attachments"] = [part.get_filename() for part in msg.get_payload() if part.get_filename()]

        def quit(self):
            sent["quit"] = True

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(st, "secrets", {
        "SMTP_SERVER": "smtp.example.org", "SMTP_PORT": 587, "SMTP_USERNAME": "sender@example.org",
        "SMTP_PASSWORD": "app-password", "SMTP_FROM_EMAIL": "sender@example.org"})
    ok, message = app._send_email_with_smtp("instructor@example.org", "Subject", "Body", [("results.zip", b"PK")])
    assert ok, message
    assert sent["host"] == "smtp.example.org" and sent["port"] == 587 and sent["tls"] is True
    assert sent["login"] == ("sender@example.org", "app-password")
    assert sent["to"] == "instructor@example.org" and sent["subject"] == "Subject"
    assert sent["attachments"] == ["results.zip"] and sent["quit"] is True

    # without any configuration the function reports it instead of raising
    monkeypatch.setattr(st, "secrets", {})
    ok, message = app._send_email_with_smtp("instructor@example.org", "Subject", "Body")
    assert not ok and "not configured" in message
