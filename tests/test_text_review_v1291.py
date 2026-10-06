"""Findings of the adversarial review of the text pipeline: structured columns stay untouched, fillers do not split phrases."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils import detect_oe_columns  # noqa: E402
from utils import text_cleanup as tc  # noqa: E402
from utils.hbs_validator import HBSValidator  # noqa: E402


def test_an_open_ended_question_named_like_a_system_column_never_makes_that_column_text():
    df = pd.DataFrame({"Gender": ["Male", "Prefer not to say"], "Age": [20, 31], "Q5": ["a b c", "d e f"]})
    assert detect_oe_columns(df, {"Gender", "Age", "Q5"}) == ["Q5"]


def test_the_validator_never_treats_a_protected_structured_box_as_a_likert_item():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"Punish_1": rng.randint(1, 8, 80), "Punish_2": rng.randint(1, 8, 80), "Punish_3": rng.randint(1, 8, 80),
                       "Punish_1_2": rng.randint(1, 8, 80)})  # a numbered numeric text box that looks like an item
    plain = HBSValidator(seed=1)._find_scale_columns(df)
    protected = HBSValidator(seed=1, protected_columns={"Punish_1_2"})._find_scale_columns(df)
    assert "Punish_1_2" not in protected and {"Punish_1", "Punish_2", "Punish_3"} <= set(protected)
    assert set(protected) <= set(plain)


def test_fillers_go_only_where_they_do_not_split_a_phrase():
    rng = random.Random(3)
    for text in ("Overall. And I think it costs too much", "I looked for a while but gave up", "It is the thing to which I object",
                 "Prices rose. However the plan helped. And that is fine", "so far it works even if it is slow"):
        for _ in range(60):
            words = text.split()
            tc.insert_filler(words, "you know", rng)
            out = " ".join(words)
            assert ".," not in out and "!," not in out and "?," not in out, out
            for bad in ("for a,", "to, you know, which", "so, you know, far", "even, you know, if", "for, you know, a while"):
                assert bad not in out, (bad, out)
            assert ". you know, And" not in out and "., you know" not in out


def test_a_stranded_preposition_is_not_trimmed_but_a_dangling_conjunction_is():
    assert tc.trim_dangling_tail("the person I voted for") == "the person I voted for"
    assert tc.trim_dangling_tail("it was fine but") == "it was fine."


def test_non_english_and_non_latin_text_is_recognised():
    assert tc.is_probably_non_english("Me parece que la política es muy importante para todos nosotros")
    assert tc.is_probably_non_english("我认为这项政策非常重要")
    assert not tc.is_probably_non_english("I think the policy is quite important for all of us")
