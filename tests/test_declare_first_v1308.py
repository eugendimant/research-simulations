"""Declare-first effect inference, per-outcome provenance, SocSim scope (v1.3.0.8).

The Stage 1 audit (2026-10-23) found three P0 defects, each reproduced here before it was fixed:

* **P0-1, polarity-blind inference.** With ``auto_effects=True`` and nothing declared, a "negative
  tone" arm lowered EVERY outcome, including an outcome on which higher means more OPPOSITION. On
  the Brader, Valentino and Suhay (2008) design the focal contrast came out at g = -0.19 against a
  human +0.46: the sign agreed in 12% of seeds. The engine knew what the condition said and nothing
  about what higher values of the outcome meant. Wording-derived effects now need the analyst to
  declare the direction (or to opt into unvalidated inference explicitly), and a declaration
  orients the keyword sign.
* **P0-2, no per-variable provenance.** Every non-user effect was exported as ``"source":
  "inferred"``, whether it came from a recalled meta-analytic row, a valence keyword or nothing but
  the label hash, and ``honesty_notice()`` had no caller outside the tests.
* **P0-3, study-wide game detection.** A Likert ``Brand_Satisfaction`` DV in a study mentioning a
  "charity donation appeal" was classified as a dictator game and its item-1 mean moved from 5.60
  to 2.81, because a context-only keyword hit scored 0.55 against a 0.5 gate.
"""
import numpy as np
import pytest

from utils import socsim_adapter as SA
from utils.enhanced_simulation_engine import (
    EnhancedSimulationEngine, _normalize_declared_polarity,
)

DEMO = {"gender_quota": 50, "age_mean": 35, "age_sd": 12}


def _scale(name, items=4, points=7, **kw):
    return dict(name=name, variable_name=name, num_items=items, scale_points=points,
                scale_min=1, scale_max=points, reverse_items=[], type="matrix", **kw)


def _brader(seed=7, n=600, **kw):
    """The audited design: tone x immigrant origin, outcome = opposition to immigration."""
    e = EnhancedSimulationEngine(
        study_title="Immigration news experiment",
        study_description=("Participants saw a news story with a negative or positive tone about a "
                           "Latino or European immigrant."),
        sample_size=n, conditions=["Negative tone / Latino", "Positive tone / European"],
        factors=[], scales=[_scale("immigr")], additional_vars=[], demographics=DEMO,
        open_ended_questions=[], seed=seed, **kw)
    e.llm_generator and e.llm_generator.disable_permanently("test")
    return e


def _focal_d(df, col="immigr_mean"):
    a = df[df.CONDITION == "Negative tone / Latino"][col].to_numpy(float)
    b = df[df.CONDITION == "Positive tone / European"][col].to_numpy(float)
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


# ---------------------------------------------------------------------------------------------
# P0-1 declare-first
# ---------------------------------------------------------------------------------------------
def test_a_negative_label_does_not_move_an_outcome_of_unknown_polarity():
    """The regression the audit asked for: wording alone must not move an undeclared outcome."""
    ds = []
    for seed in range(101, 109):
        df, md = _brader(seed=seed, n=400).generate()
        ds.append(_focal_d(df))
        row = md["calibration_provenance"]["variables"][0]
        assert row["effect_source"] == "not_inferred"
        assert row["polarity"] is None and row["polarity_source"] == "undeclared"
    # a true null: the realised d is sampling noise around zero, not a confident -0.19
    assert abs(float(np.mean(ds))) < 0.08, ds
    assert max(abs(d) for d in ds) < 0.25, ds


def test_the_undeclared_run_says_so_instead_of_handing_back_a_quiet_null():
    _df, md = _brader().generate()
    warns = " ".join(md["generation_warnings"])
    assert "No effect was inferred for immigr" in warns
    assert "higher" in warns.lower()


def test_declared_reverse_polarity_puts_the_sign_the_right_way_round():
    """Higher = more opposition, so a negative-tone arm must score HIGHER, not lower."""
    d_rev = _focal_d(_brader(dv_polarity={"immigr": "-1"}).generate()[0])
    d_fwd = _focal_d(_brader(dv_polarity={"immigr": "+1"}).generate()[0])
    assert d_rev > 0.15, d_rev
    assert d_fwd < -0.15, d_fwd
    assert d_rev == pytest.approx(-d_fwd, abs=0.02)


def test_a_free_text_declaration_is_read_and_recorded_as_a_label():
    df, md = _brader(dv_polarity={"immigr": "more opposition to immigration"}).generate()
    row = md["calibration_provenance"]["variables"][0]
    assert row["polarity"] == "higher_is_less" and row["polarity_source"] == "declared_label"
    assert _focal_d(df) > 0.15


@pytest.mark.parametrize("value, expected", [
    ("+1", 1), ("positive", 1), (1, 1), (True, 1), ("higher_is_more", 1),
    ("-1", -1), ("negative", -1), (-2.0, -1), ("reverse", -1), ("higher_is_worse", -1),
    ("", None), (None, None), ("purple", None), (0, None),
    ("more support for the policy", 1), ("greater distress", -1),
])
def test_polarity_declarations_are_read_as_written(value, expected):
    assert _normalize_declared_polarity(value)[0] == expected


def test_opting_in_restores_wording_inference_and_labels_it_a_guess():
    df, md = _brader(infer_effects_without_polarity=True).generate()
    row = md["calibration_provenance"]["variables"][0]
    assert row["effect_source"] == "keyword_rule"
    assert "not validated" in row["status"]
    assert md["calibration_provenance"]["wording_inference_enabled"] is True
    assert _focal_d(df) < -0.1          # the old, unvalidated behaviour, unchanged
    assert not any("No effect was inferred" in w for w in md["generation_warnings"])


def test_a_declared_polarity_is_enough_without_the_opt_in():
    _df, md = _brader(dv_polarity={"immigr": "-1"}).generate()
    assert md["calibration_provenance"]["declare_first"] is True
    assert md["calibration_provenance"]["variables"][0]["effect_source"] == "keyword_rule"


def test_a_user_specified_effect_is_never_gated_by_the_polarity_rule():
    """Declare-first governs INFERENCE. An effect the analyst specified is theirs and applies."""
    e = _brader(effect_sizes=[{"variable": "immigr", "factor": "CONDITION",
                               "level_high": "Negative tone / Latino",
                               "level_low": "Positive tone / European",
                               "cohens_d": 0.5, "direction": "positive"}])
    df, md = e.generate()
    assert _focal_d(df) > 0.3
    row = md["calibration_provenance"]["variables"][0]
    assert row["effect_source"] == "user"
    assert not any("No effect was inferred" in w for w in md["generation_warnings"])


def test_an_economic_game_outcome_keeps_its_direction_from_the_scale():
    """An allocation DV needs no declaration: more money sent IS more money sent."""
    e = EnhancedSimulationEngine(
        study_title="Dictator game", study_description="Dictators split $100 with an ingroup or outgroup recipient.",
        sample_size=600, conditions=["Ingroup recipient", "Outgroup recipient"], factors=[],
        scales=[dict(name="Give", variable_name="Give", num_items=1, scale_points=101,
                     scale_min=0, scale_max=100, reverse_items=[], type="slider")],
        additional_vars=[], demographics=DEMO, open_ended_questions=[], seed=3)
    e.llm_generator and e.llm_generator.disable_permanently("test")
    df, md = e.generate()
    row = next(r for r in md["calibration_provenance"]["variables"] if r["variable"] == "Give")
    assert row["polarity_source"] == "game_geometry"
    assert row["effect_source"] != "not_inferred"
    a = df[df.CONDITION == "Ingroup recipient"]["Give_1"].to_numpy(float)
    b = df[df.CONDITION == "Outgroup recipient"]["Give_1"].to_numpy(float)
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    assert d > 0.2, f"ingroup favouritism was flattened: d = {d:.3f}"


def test_auto_effects_false_is_still_a_true_null_and_says_none():
    _df, md = _brader(auto_effects=False).generate()
    assert md["calibration_provenance"]["variables"][0]["effect_source"] == "none"
    assert md["effect_sizes_applied"]["inferred_effects_enabled"] is False


# ---------------------------------------------------------------------------------------------
# P0-2 provenance export
# ---------------------------------------------------------------------------------------------
_FINER = {"user", "curated_paradigm", "paradigm_anchor", "literature_fallback", "keyword_rule",
          "hash_residue", "reference_arm", "not_inferred", "none"}


def test_no_contrast_is_exported_as_inferred_without_a_finer_label():
    for kwargs in ({}, {"infer_effects_without_polarity": True},
                   {"dv_polarity": {"immigr": "-1"}}, {"auto_effects": False}):
        _df, md = _brader(**kwargs).generate()
        rows = md["effect_sizes_applied"]["contrasts"]
        assert rows
        for r in rows:
            assert r.get("effect_source") in _FINER, r
            assert r.get("status")
            if r["source"] == "inferred":
                assert r["effect_source"] != "inferred"


def test_a_novel_design_reports_an_uncalibrated_default_for_every_outcome():
    e = EnhancedSimulationEngine(
        study_title="Zorblax study", study_description="Participants rated a zorblax under two framings.",
        sample_size=200, conditions=["Framing qux", "Framing zib"], factors=[],
        scales=[_scale("Zorblax_rating"), _scale("Quux_index", items=3)],
        additional_vars=[], demographics=DEMO, open_ended_questions=[], seed=5)
    e.llm_generator and e.llm_generator.disable_permanently("test")
    _df, md = e.generate()
    rows = md["calibration_provenance"]["variables"]
    assert {r["variable"] for r in rows} == {"Zorblax_rating", "Quux_index"}
    for r in rows:
        assert r["effect_source"] == "not_inferred"
        assert "uncalibrated default" in r["status"]
        assert r["marginal_source"] in ("generic_default", "construct_keyword")


def test_the_provenance_table_carries_the_registry_verification_notice():
    _df, md = _brader().generate()
    prov = md["calibration_provenance"]
    assert "NOT been checked" in prov["registry_notice"]
    assert prov["registry_coverage"]["total_entries"] > 100


def test_the_marginal_source_of_a_game_outcome_names_the_published_row():
    e = EnhancedSimulationEngine(
        study_title="Dictator game", study_description="Dictators split $100 between themselves and a recipient.",
        sample_size=300, conditions=["A", "B"], factors=[],
        scales=[dict(name="Allocation", variable_name="Allocation", num_items=1, scale_points=101,
                     scale_min=0, scale_max=100, reverse_items=[], type="slider")],
        additional_vars=[], demographics=DEMO, open_ended_questions=[], seed=9)
    e.llm_generator and e.llm_generator.disable_permanently("test")
    _df, md = e.generate()
    row = md["calibration_provenance"]["variables"][0]
    assert row["marginal_source"] == "game_row"
    assert row["marginal_reference"]


def test_the_instructor_report_shows_the_provenance_table():
    from utils.instructor_report import ComprehensiveInstructorReport
    df, md = _brader(n=200).generate()
    gen = ComprehensiveInstructorReport()
    rows = gen._provenance_rows(md)
    assert rows and rows[0]["variable"] == "immigr"
    md_lines = "\n".join(gen._provenance_md(md))
    assert "Where each outcome's numbers came from" in md_lines
    assert "not declared" in md_lines
    assert "NOT been checked" in md_lines
    html = "".join(gen._provenance_html(md))
    assert "<table>" in html and "immigr" in html


# ---------------------------------------------------------------------------------------------
# P0-3 SocSim scope
# ---------------------------------------------------------------------------------------------
def test_a_likert_dv_in_a_donation_study_is_not_a_dictator_game():
    """Probe (a) from the audit: study context no longer makes an unrelated DV a game DV."""
    likert = _scale("Brand_Satisfaction", items=5)
    hits = SA.detect_game_dvs(
        scales=[likert], study_title="Charity donation appeal",
        study_description="Participants saw a charity donation appeal and rated the brand.",
        conditions=["Appeal A", "Appeal B"])
    assert hits == []


def test_the_likert_dv_is_left_exactly_where_the_engine_put_it():
    def run(use_socsim):
        e = EnhancedSimulationEngine(
            study_title="Charity donation appeal",
            study_description="Participants saw a charity donation appeal and rated the brand.",
            sample_size=300, conditions=["Appeal A", "Appeal B"], factors=[],
            scales=[_scale("Brand_Satisfaction", items=5)], additional_vars=[], demographics=DEMO,
            open_ended_questions=[], seed=5, use_socsim_experimental=use_socsim)
        e.llm_generator and e.llm_generator.disable_permanently("test")
        return e.generate()
    off_df, _ = run(False)
    on_df, on_md = run(True)
    assert (on_md.get("socsim") or {}).get("reason") == "no_game_dvs_detected"
    assert on_df["Brand_Satisfaction_1"].mean() == pytest.approx(
        off_df["Brand_Satisfaction_1"].mean(), abs=1e-9)


@pytest.mark.parametrize("name, items, lo, hi, kind, fits", [
    ("Dictator_Allocation", 1, 0, 100, "slider", True),      # a real allocation
    ("Dictator_Giving", 3, 1, 7, "matrix", False),           # a 7-point attitude matrix
    ("Donation_Attitude", 1, 1, 7, "single_item", False),     # single item, but Likert geometry
    ("Contribution_to_Society", 4, 1, 7, "matrix", False),
])
def test_geometry_must_fit_the_game(name, items, lo, hi, kind, fits):
    scale = dict(name=name, variable_name=name, num_items=items, scale_points=hi - lo + 1,
                 scale_min=lo, scale_max=hi, reverse_items=[], type=kind)
    assert bool(SA.detect_game_dvs(scales=[scale])) is fits


def test_detection_records_its_evidence():
    scale = dict(name="Dictator_Allocation", variable_name="Dictator_Allocation", num_items=1,
                 scale_points=101, scale_min=0, scale_max=100, reverse_items=[], type="slider")
    hit = SA.detect_game_dvs(scales=[scale])[0]
    assert hit["game_name"] == "dictator" and hit["confidence"] == 0.85
    assert hit["evidence"]["matched_in"] == "dv_own_text"
    assert "allocation scale" in hit["evidence"]["geometry"]
    assert hit["evidence"]["matched_pattern"]


def test_the_game_model_no_longer_flattens_the_engines_own_contrast():
    """Probe (b): with SocSim on, the knowledge-base ingroup/outgroup effect must survive."""
    def run(use_socsim, seed=3):
        e = EnhancedSimulationEngine(
            study_title="Dictator game",
            study_description="Dictators split $100 with an ingroup or outgroup recipient.",
            sample_size=600, conditions=["Ingroup recipient", "Outgroup recipient"], factors=[],
            scales=[dict(name="Dictator_Allocation", variable_name="Dictator_Allocation",
                         num_items=1, scale_points=101, scale_min=0, scale_max=100,
                         reverse_items=[], type="slider")],
            additional_vars=[], demographics=DEMO, open_ended_questions=[], seed=seed,
            use_socsim_experimental=use_socsim)
        e.llm_generator and e.llm_generator.disable_permanently("test")
        df, md = e.generate()
        a = df[df.CONDITION == "Ingroup recipient"]["Dictator_Allocation_1"].to_numpy(float)
        b = df[df.CONDITION == "Outgroup recipient"]["Dictator_Allocation_1"].to_numpy(float)
        return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2), md
    d_off, _ = run(False)
    d_on, md_on = run(True)
    assert (md_on.get("socsim") or {}).get("enriched_dvs"), "the game model did not run"
    assert md_on["socsim"]["detection_evidence"][0]["game"] == "dictator"
    assert d_off > 0.2, f"KB path lost its own effect: {d_off:.3f}"
    assert d_on > 0.2, f"game model flattened the KB effect: {d_on:.3f} (was -0.065)"


def test_the_app_does_not_switch_the_experimental_game_model_on_by_itself():
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / "simulation_app" / "app.py"
    text = src.read_text(encoding="utf-8")
    assert '"_use_socsim_experimental"] = True' not in text, (
        "the experimental game model must be an explicit opt-in, never forced on")
