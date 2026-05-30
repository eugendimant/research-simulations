#!/usr/bin/env python3
"""Fuzz _get_automatic_condition_effect across diverse condition strings and
domain contexts to flush out UnboundLocalError-style scoping bugs (variables
defined inside one branch but referenced in another). Run: python3 tests/effect_fuzz.py
"""
import os, sys, itertools, traceback
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
import warnings; warnings.filterwarnings("ignore")
from utils.enhanced_simulation_engine import EnhancedSimulationEngine

# Different study contexts → different self.detected_domains → different branches
CONTEXTS = [
    ("Political Dictator Game", "Partisan identity and economic allocation in a dictator game with Trump supporters."),
    ("AI Trust Study", "Effect of AI-generated vs human content on consumer trust and purchase intention."),
    ("Moral Judgment", "Trolley-style moral dilemmas, utilitarian vs deontological framing."),
    ("Public Goods Game", "Cooperation and contribution in a repeated public goods game with punishment."),
    ("Charity Nudge", "Default nudges and social norms on charitable giving and donations."),
    ("Generic Survey", "A general attitudes survey."),
]

# Condition strings exercising many branches, incl. relational × factorial combos
CONDS = [
    "", "control", "baseline", "treatment",
    # valence
    "positive feedback", "negative feedback", "high reward", "low reward", "neutral",
    # relational political (sets _handled_by_relational)
    "Trump supporter", "Trump hater", "Trump lover", "Trump supporter and fan",
    "Biden voter", "ingroup partner", "outgroup partner", "same party", "opposing party",
    "no identity control",
    # relational × factorial (the crash class)
    "Trump supporter × high stakes", "Trump hater x loss frame", "ingroup + gain",
    "outgroup partner & unfair", "ingroup member / hedonic", "Trump lover * utilitarian",
    "outgroup × Trump hater", "same race x high reward",
    # factorial non-relational
    "AI × Hedonic", "No AI x Utilitarian", "loss frame + high anchor",
    "fair × gain", "unfair x loss", "human / practical",
    # economic-game phrasings
    "dictator high endowment", "trust game partner", "ultimatum low offer",
    "public goods high multiplier", "punishment condition",
    # weird/edge
    "a", "x", "×", " + ", "VERY LONG CONDITION NAME " * 8, "Conditión with áccents",
    "lover", "hater", "supporter and admirer", "critic and detractor",
]

VARS = ["dollars_allocated", "trust_rating", "purchase_intention", "amount_sent",
        "contribution", "satisfaction", "moral_acceptability", "die_roll_report", "x"]

errors = []
n = 0
for (title, desc) in CONTEXTS:
    try:
        eng = EnhancedSimulationEngine(
            study_title=title, study_description=desc, sample_size=20,
            conditions=[{"name": "c1"}, {"name": "c2"}], factors=[],
            scales=[{"name": "dv", "items": 1, "min": 0, "max": 100}], additional_vars=[],
            demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, seed=3)
    except Exception as e:
        errors.append((title, "<init>", "<->", f"{type(e).__name__}: {e}"))
        traceback.print_exc(); continue
    for cond, var in itertools.product(CONDS, VARS):
        n += 1
        try:
            v = eng._get_automatic_condition_effect(cond, var)
            # result must be a finite float
            if not isinstance(v, (int, float)):
                errors.append((title, cond, var, f"non-numeric result: {v!r}"))
        except Exception as e:
            errors.append((title, cond, var, f"{type(e).__name__}: {e}"))

print(f"Fuzzed {n} (context × condition × variable) combinations\n")
if errors:
    print(f"FAIL: {len(errors)} errors")
    # de-dup by error message
    seen = set()
    for title, cond, var, msg in errors:
        key = msg.split(":")[0] + "|" + cond[:30]
        if key in seen:
            continue
        seen.add(key)
        print(f"  [{title}] cond={cond!r:35.35} var={var!r:18} -> {msg}")
    sys.exit(1)
print("PASS: no exceptions, all results numeric")
sys.exit(0)
