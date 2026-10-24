"""P1 defects from the Stage 1 audit (2026-10-23), reproduced and fixed in v1.3.0.9.

* **P1-4, numeric DV collapse.** A numeric DV declared 60-80 returned 60 for all 400 participants.
  The validator read the range from ``scale_points`` instead of ``scale_max``, so every draw was
  clipped to the floor. ``skew()`` of a constant column is 0.0 on pandas 3, which is why the
  neighbouring test passed while the column was dead.
* **P1-5, binary marginals.** With effects off the engine's binary base rate sat near 0.62 whatever
  the study (human 0.11 on the audited design: a mean error of 1.63 human SDs). The rate is now the
  analyst's to declare; undeclared it is an explicit, flagged 0.50, never a hidden constant.
* **P1-6, parser structure.** A 2 x 2 run as two sibling randomizers came through as ONE factor with
  four levels, and binary single-choice outcomes were not detected at all, so the audited design
  lost both of its behavioural DVs. Zero-outcome surveys got a fabricated 5-item 7-point "Main_DV"
  and a page announcing "1 DV(s) detected".
* **P1-7, tier damping.** Every tier was damped by the same 0.45, the anchor path passed ``key=""``
  so no tier was ever looked up, and the 19 rule-only paradigms were discounted twice.
* **P1-8, mis-routed calibrations.** ``Willingness_to_Share`` / ``Payment_Satisfaction`` /
  ``Offer_Attractiveness`` were read as economic allocations; ``Behavior_Score`` matched the
  social-comparison norm through the fragment "sco"; ``Sleep_Quality`` got the PSQI clinical norm.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.enhanced_simulation_engine import EnhancedSimulationEngine  # noqa: E402
from utils.qsf_preview import QSFPreviewParser  # noqa: E402

DEMO = {"gender_quota": 50, "age_mean": 35, "age_sd": 12}


def _engine(scales, n=400, seed=11, conditions=None, **kw):
    e = EnhancedSimulationEngine(
        study_title="Numeric and binary outcomes", study_description="Audit regression fixture.",
        sample_size=n, conditions=conditions or ["Control", "Treatment"], factors=[],
        scales=scales, additional_vars=[], demographics=DEMO, open_ended_questions=[],
        seed=seed, auto_effects=False, **kw)
    if e.llm_generator:
        e.llm_generator.disable_permanently("test")
    return e


# -------------------------------------------------------------------------------------------------
# P1-4 numeric DV range
# -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lo,hi", [(60, 80), (-10, 10), (0, 1000), (1, 7), (0, 100)])
def test_a_numeric_dv_spans_its_declared_range(lo, hi):
    """The audit's acceptance criterion: non-constant data with a plausible mean for each range."""
    spec = dict(name="Amount", variable_name="Amount", num_items=1, type="numeric_input",
                scale_min=lo, scale_max=hi, scale_points=max(2, min(int(hi - lo) + 1, 100)),
                reverse_items=[])
    df, _ = _engine([spec]).generate()
    col = df["Amount"] if "Amount" in df.columns else df["Amount_1"]
    vals = col.to_numpy(float)
    assert vals.std(ddof=1) > 0, f"{lo}-{hi} came back constant at {vals[0]}"
    assert vals.min() >= lo and vals.max() <= hi
    # the mass sits inside the range rather than piled on an endpoint
    assert (vals == lo).mean() < 0.5 and (vals == hi).mean() < 0.5
    assert lo < vals.mean() < hi


def test_the_audited_60_to_80_case_is_not_pinned_to_its_floor():
    spec = dict(name="Amount", variable_name="Amount", num_items=1, type="numeric_input",
                scale_min=60, scale_max=80, scale_points=21, reverse_items=[])
    df = _engine([spec], n=400).generate()[0]
    vals = df[next(c for c in df.columns if c.startswith("Amount"))].to_numpy(float)
    assert len(set(vals.tolist())) > 5, "fewer than six distinct values: the old collapse"
    assert vals.std(ddof=1) > 1.0


# -------------------------------------------------------------------------------------------------
# P1-5 binary base rate is declared, never hidden
# -------------------------------------------------------------------------------------------------
def _binary_spec(name="Signed", rate=None):
    spec = dict(name=name, variable_name=name, num_items=1, type="binary", scale_min=0,
                scale_max=1, scale_points=2, reverse_items=[])
    if rate is not None:
        spec["base_rate"] = rate
    return spec


@pytest.mark.parametrize("rate", [0.11, 0.5, 0.8])
def test_a_declared_binary_base_rate_is_recovered(rate):
    df, _ = _engine([_binary_spec(rate=rate)], n=1200, seed=5).generate()
    col = "Signed" if "Signed" in df.columns else "Signed_1"
    got = df[col].to_numpy(float).mean()
    assert abs(got - rate) < 0.06, f"declared {rate}, got {got:.3f}"


def test_an_undeclared_binary_rate_is_an_explicit_uninformed_half():
    e = _engine([_binary_spec()], n=1200, seed=5)
    df, meta = e.generate()
    col = "Signed" if "Signed" in df.columns else "Signed_1"
    got = df[col].to_numpy(float).mean()
    assert abs(got - 0.5) < 0.06, f"undeclared rate should be an uninformed 0.50, got {got:.3f}"
    warnings = " ".join(str(w) for w in (meta.get("generation_warnings") or []))
    assert "base rate" in warnings.lower(), "an undeclared binary DV must say so"


# -------------------------------------------------------------------------------------------------
# P1-6 parser: crossed factors and binary outcomes
# -------------------------------------------------------------------------------------------------
def _sq(qid, tag, text, qtype, selector, **extra):
    payload = {"QuestionText": text, "DataExportTag": tag, "QuestionType": qtype, "Selector": selector,
               "QuestionDescription": text, "QuestionID": qid, "Language": [],
               "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}}}
    payload.update(extra)
    return {"SurveyID": "SV_t", "Element": "SQ", "PrimaryAttribute": qid, "SecondaryAttribute": text,
            "TertiaryAttribute": None, "Payload": payload}


def _likert(qid, tag, text):
    return _sq(qid, tag, text, "MC", "SAVR",
               Choices={str(i): {"Display": d} for i, d in enumerate(
                   ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"], 1)},
               ChoiceOrder=[1, 2, 3, 4, 5])


def _binary_q(qid, tag, text):
    return _sq(qid, tag, text, "MC", "SAVR",
               Choices={"1": {"Display": "Yes"}, "2": {"Display": "No"}}, ChoiceOrder=[1, 2])


def _qsf(block_names, questions, flow, name="Synthetic"):
    blocks = {}
    for i, (bid, (desc, qids)) in enumerate(block_names.items(), 1):
        blocks[str(i)] = {"Type": "Standard", "Description": desc, "ID": bid,
                          "BlockElements": [{"Type": "Question", "QuestionID": q} for q in qids]}
    els = list(questions)
    els.append({"SurveyID": "SV_t", "Element": "BL", "PrimaryAttribute": "Survey Blocks",
                "SecondaryAttribute": None, "TertiaryAttribute": None, "Payload": blocks})
    els.append({"SurveyID": "SV_t", "Element": "FL", "PrimaryAttribute": "Survey Flow",
                "SecondaryAttribute": None, "TertiaryAttribute": None,
                "Payload": {"Type": "Root", "FlowID": "FL_1", "Flow": flow}})
    return json.dumps({"SurveyEntry": {"SurveyID": "SV_t", "SurveyName": name, "SurveyDescription": "",
                                       "SurveyLanguage": "EN"}, "SurveyElements": els}).encode("utf-8")


def _blk(bid):
    return {"Type": "Standard", "ID": bid, "FlowID": f"FL_{bid}"}


def _brader_qsf():
    """The audited design: emotional tone x immigrant origin, three outcomes, two of them binary."""
    qs = [_sq("QID1", "intro", "News story about immigration.", "DB", "TB"),
          _sq("QID10", "tone_anx", "The story stresses the costs of immigration.", "DB", "TB"),
          _sq("QID11", "tone_ent", "The story stresses the benefits of immigration.", "DB", "TB"),
          _sq("QID20", "cue_lat", "The immigrant pictured is from Mexico.", "DB", "TB"),
          _sq("QID21", "cue_eur", "The immigrant pictured is from Russia.", "DB", "TB"),
          _likert("QID30", "immigr", "Should the number of immigrants permitted to come to the "
                                     "United States be increased or decreased?"),
          _binary_q("QID31", "cong_mesg", "Would you like us to send a message to Congress about "
                                          "immigration on your behalf?"),
          _binary_q("QID32", "anti_info", "Would you like to request more information from an "
                                          "anti-immigration group?")]
    blocks = {"BL_i": ("Introduction", ["QID1"]),
              "BL_anx": ("Anxiety", ["QID10"]), "BL_ent": ("Enthusiasm", ["QID11"]),
              "BL_lat": ("Latino Cue", ["QID20"]), "BL_eur": ("European Cue", ["QID21"]),
              "BL_dv": ("Outcomes", ["QID30", "QID31", "QID32"])}
    flow = [_blk("BL_i"),
            {"Type": "BlockRandomizer", "FlowID": "FL_r1", "SubSet": "1", "EvenPresentation": True,
             "Flow": [_blk("BL_anx"), _blk("BL_ent")]},
            {"Type": "BlockRandomizer", "FlowID": "FL_r2", "SubSet": "1", "EvenPresentation": True,
             "Flow": [_blk("BL_lat"), _blk("BL_eur")]},
            _blk("BL_dv")]
    return _qsf(blocks, qs, flow, name="Brader Valentino Suhay 2008 (synthetic)")


def test_two_sibling_randomizers_are_read_as_a_crossed_design():
    pv = QSFPreviewParser().parse(_brader_qsf())
    assert [len(f["levels"]) for f in pv.detected_factors] == [2, 2], pv.detected_factors
    assert len(pv.suggested_crossed_conditions) == 4
    # the flat arm list is left alone, so nothing downstream changes until the reading is adopted
    assert pv.detected_conditions == ["Anxiety", "Enthusiasm", "Latino Cue", "European Cue"]


def test_the_audited_design_recovers_two_by_two_and_three_dvs_without_manual_addition():
    import app as appmod
    inferred = appmod._preview_to_engine_inputs(QSFPreviewParser().parse(_brader_qsf()))
    assert inferred["design_reading"] == "crossed_from_flow"
    assert len(inferred["conditions"]) == 4
    assert [len(f["levels"]) for f in inferred["factors"]] == [2, 2]
    assert not inferred["no_outcome_detected"]
    names = [s.get("variable_name") for s in inferred["scales"]]
    assert set(names) == {"immigr", "cong_mesg", "anti_info"}, names
    binaries = [s["variable_name"] for s in inferred["scales"] if s.get("type") == "binary"]
    assert set(binaries) == {"cong_mesg", "anti_info"}, binaries
    for s in inferred["scales"]:
        if s["variable_name"] in binaries:
            assert (s["scale_min"], s["scale_max"], s["scale_points"]) == (0, 1, 2)


def test_label_splitting_alone_could_not_see_the_crossing():
    """Why the structural read is needed: these four arm labels share no separator."""
    import app as appmod
    label_only = appmod._infer_factors_from_conditions(
        ["Anxiety", "Enthusiasm", "Latino Cue", "European Cue"])
    assert len(label_only) == 1, label_only


def test_a_survey_with_no_outcome_is_flagged_rather_than_given_a_fabricated_dv():
    import app as appmod
    qs = [_sq("QID1", "consent", "Do you consent to take part?", "DB", "TB")]
    blocks = {"BL_c": ("Consent", ["QID1"])}
    inferred = appmod._preview_to_engine_inputs(
        QSFPreviewParser().parse(_qsf(blocks, qs, [_blk("BL_c")])))
    assert inferred["no_outcome_detected"] is True
    assert inferred["scales"][0]["_is_placeholder"] is True
    assert inferred["scales"][0]["detected_from_qsf"] is False


def test_the_design_page_says_no_outcome_was_found():
    src = (_APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "No outcome variable was found in this survey" in src
    assert "_qsf_design_scales(preview)" in src


# -------------------------------------------------------------------------------------------------
# P1-7 tier damping
# -------------------------------------------------------------------------------------------------
def test_the_tier_weight_scales_the_floor_of_the_retained_effect():
    """The tier was looked up with key="" on the anchor path, so no tier ever reached the floor."""
    from utils import empirical_registry as ER
    pol = ER.EffectPolicy(mode="replication_adjusted", shrinkage=None, heterogeneity_draw=False)
    best = max(ER.TIER_WEIGHT, key=lambda t: ER.TIER_WEIGHT[t])
    worst = min(ER.TIER_WEIGHT, key=lambda t: ER.TIER_WEIGHT[t])
    assert ER.TIER_WEIGHT[best] > ER.TIER_WEIGHT[worst]
    floors = [abs(0.8) * pol.min_retained * ER.TIER_WEIGHT[t] for t in (best, worst)]
    assert floors[0] > floors[1], "a better-verified entry must keep more of its effect"
    got = ER.adjust_effect(0.8, kind="meta", key="", policy=pol)
    assert np.isfinite(got) and abs(got) <= 0.8


def test_the_registry_reports_honestly_whether_the_tier_changes_the_magnitude():
    from utils import empirical_registry as ER
    info = ER.tier_cap_binds(ER.EffectPolicy(mode="replication_adjusted", shrinkage=None))
    assert set(info) >= {"shrinkage_applied", "tiers_that_cap_the_magnitude",
                         "magnitude_is_tier_sensitive", "note"}
    # the claim must match the measurement, whichever way it comes out
    assert bool(info["tiers_that_cap_the_magnitude"]) == bool(info["magnitude_is_tier_sensitive"])


@pytest.mark.parametrize("label", ["social norm", "Mortality salience"])
def test_a_rule_only_paradigm_is_shrunk_once_not_twice(label):
    """The 19 rule-only paradigms were damped by their tier inside _meta_index and AGAIN by the
    policy, so a published d of 0.47 was built in at 0.32."""
    from utils import empirical_registry as ER
    from utils import literature_effects as LE
    m = LE._rule_lookup(label)
    assert m is not None, label
    assert m.rule == "label_phrase"
    assert m.shrinkage == 1.0, "the match must not carry a second discount of its own"
    # exactly one policy factor stands between the published and the applied magnitude
    f = ER.policy_factor()
    assert abs(abs(m.effect_d) - abs(m.published_d) * f) < 1e-6, (m.effect_d, m.published_d, f)


def test_a_colour_label_pair_is_not_a_paradigm_match():
    """The audit's token mis-routing: "Blue banner" vs "Green banner" hit green_marketing_meta."""
    from utils import literature_effects as LE
    for a, b in (("Blue banner", "Green banner"), ("Green button", "Blue button")):
        assert LE.lookup(a, b, "website layout study") is None, (a, b)


# -------------------------------------------------------------------------------------------------
# P1-8 calibration routing
# -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["Willingness_to_Share", "Payment_Satisfaction",
                                  "Offer_Attractiveness", "Behavior_Score", "Sleep_Quality",
                                  "Hopelessness"])
def test_an_unmatched_variable_falls_through_to_a_flagged_generic_default(name):
    e = _engine([dict(name=name, variable_name=name, num_items=3, scale_points=7, scale_min=1,
                      scale_max=7, reverse_items=[], type="matrix")], n=200)
    _, meta = e.generate()
    prov = _prov(meta, name)
    assert prov, f"no provenance exported for {name}"
    # either nothing matched, or a flagged generic keyword branch -- never a specific published
    # norm or game row that is about something else
    assert prov["marginal_source"] in ("generic_default", "construct_keyword"), prov
    ref = str(prov["marginal_reference"]).lower()
    for wrong in ("allocation", "economic", "dictator", "psqi", "social_comparison", "helping"):
        assert wrong not in ref, (name, prov)


@pytest.mark.parametrize("name", ["PSS4_1", "bfi_extra_1", "stress_total"])
def test_a_variable_that_names_its_construct_still_reaches_its_norm(name):
    e = _engine([dict(name=name, variable_name=name, num_items=4, scale_points=5, scale_min=1,
                      scale_max=5, reverse_items=[], type="matrix")], n=200)
    _, meta = e.generate()
    prov = _prov(meta, name)
    assert prov.get("marginal_source") == "construct_norm", prov


def _prov(meta, name):
    rows = ((meta.get("calibration_provenance") or {}).get("variables") or [])
    return next((r for r in rows if r.get("variable") == name), {})


def test_a_deficit_scored_norm_needs_a_declared_direction():
    """PSQI's score runs 0 (slept well) to 21 (slept badly): on an item called Sleep_Quality its
    published mean put the baseline at the bad end of a scale on which higher means better."""
    base = dict(name="Sleep_Quality", variable_name="Sleep_Quality", num_items=3, scale_points=7,
                scale_min=1, scale_max=7, reverse_items=[], type="matrix")
    _, meta = _engine([dict(base)], n=200).generate()
    assert _prov(meta, "Sleep_Quality")["marginal_source"] == "generic_default"
    declared = dict(base, higher_means="+1")
    _, meta2 = _engine([declared], n=200).generate()
    row = _prov(meta2, "Sleep_Quality")
    assert row["marginal_source"] == "construct_norm" and "psqi" in row["marginal_reference"].lower()
    assert row["polarity"] == "higher_is_more"


def test_one_generic_word_is_not_a_construct_match():
    from utils import construct_matcher as CM
    for v in ("Behavior_Score", "Brand_Satisfaction", "Attitude_Rating", "Trust_Score"):
        assert CM.match(variable_name=v) is None, v
    # the instruments the matcher exists for still resolve
    for v, key in (("PSS4_1", "perceived_stress_pss"), ("bfi_extra_1", "big_five_extraversion"),
                   ("stress_total", "perceived_stress_pss")):
        m = CM.match(variable_name=v)
        assert m is not None and m.key == key, (v, m)


def test_every_construct_map_target_exists():
    """29 of the 103 keyword targets pointed at norms that are not in the table: a hit on one of
    them silently produced nothing, so the variable looked calibrated and was not."""
    import ast

    from utils import scientific_knowledge_base as KB
    src = (_APP_DIR / "utils" / "enhanced_simulation_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    targets: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_construct_map" for t in node.targets):
            for val in getattr(node.value, "values", []):
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    targets.add(val.value)
    assert len(targets) > 30, f"only found {len(targets)} targets: the parse missed the table"
    missing = sorted(t for t in targets if t not in KB.CONSTRUCT_NORMS)
    assert not missing, missing
