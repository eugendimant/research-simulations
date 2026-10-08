"""Guards for the v1.3.0.6 evidence pass.

Two data files back it:

* ``registry/evidence_v1306.json`` - MEASURED entries computed by
  ``tools/derive_evidence_benchmarks.py`` from open datasets we actually held. The
  raw files are not committed, so the tests recompute every aggregate from the
  committed block profiles, and re-run the script end to end when
  ``EVIDENCE_DATA_ROOT`` points at the downloaded files.
* ``registry/corroboration_v1306.json`` - what web-search summaries state about the
  numbers the engine leans on. It carries verdicts, never a tier.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "simulation_app"))

from utils import empirical_registry as reg  # noqa: E402

REG_DIR = os.path.join(ROOT, "simulation_app", "utils", "registry")
EVID = os.path.join(REG_DIR, "evidence_v1306.json")
CORR = os.path.join(REG_DIR, "corroboration_v1306.json")
TOOL = os.path.join(ROOT, "tools", "derive_evidence_benchmarks.py")


@pytest.fixture(scope="module")
def evidence():
    with open(EVID, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def corroboration():
    with open(CORR, encoding="utf-8") as fh:
        return json.load(fh)


def _tool():
    spec = importlib.util.spec_from_file_location("derive_evidence_benchmarks", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- evidence
def test_evidence_entries_load_as_measured(evidence):
    reg.reload_entries()
    loaded = {k: v for k, v in reg.entries().items() if k.startswith("evidence.")}
    assert len(loaded) == len(evidence["entries"]) >= 90
    for ent in loaded.values():
        assert ent.tier == reg.MEASURED, ent.entry_id
        assert ent.may_set_magnitude


def test_every_entry_names_its_script_and_hashed_sources(evidence):
    for e in evidence["entries"]:
        prov = e["provenance"]
        assert prov["script"] == "tools/derive_evidence_benchmarks.py"
        assert os.path.exists(os.path.join(ROOT, prov["script"]))
        assert prov["retrieved"] == "2026-10-08"
        assert prov["sources"], e["entry_id"]
        for s in prov["sources"]:
            assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"]), (e["entry_id"], s["dataset"])
            assert s["retrieved_from"].startswith("https://")
            assert s["bytes"] > 0
        assert prov["consumed_by_engine"] is False


def test_evidence_prefix_is_not_consulted_and_other_scales_keep_their_cells(evidence):
    """The `evidence.` ids are not read by any engine path; the engine-facing 7-point cells
    are scoped to seven options, so 5- and 9-point lookups still return the old cells."""
    assert all(e["entry_id"].startswith("evidence.") for e in evidence["entries"])
    for sp in (5, 9):
        sig = {"scale_points": sp, "items_per_block": 5, "keying": "same"}
        hit = reg.lookup_best("item.likert.same.k5", "straightlined_share", sig)
        assert hit is not None and hit.entry_id == "item.likert.same.k5.straightlined_share"
    hit = reg.lookup_best("item.likert.any", "item_sd_fraction_of_span", {"items_per_block": 20})
    assert hit is not None and hit.entry_id == "item.likert.any.item_sd_fraction_of_span"


def test_seven_point_blocks_get_the_measured_seven_point_cells():
    for keying, k, lo, hi in (("same", 5, 0.02, 0.045), ("mixed", 5, 0.002, 0.01), ("same", 3, 0.08, 0.15),
                              ("mixed", 8, 0.0003, 0.003), ("same", 9, 0.001, 0.01)):
        hit = reg.lookup_best(f"item.likert.{keying}.k{k}", "straightlined_share",
                              {"scale_points": 7, "items_per_block": k, "keying": keying})
        assert hit is not None and ".sp7." in hit.entry_id, (keying, k)
        assert lo < hit.value < hi, (keying, k, hit.value)
    five = reg.lookup_best("item.likert.same.k5", "straightlined_share",
                           {"scale_points": 5, "items_per_block": 5, "keying": "same"})
    seven = reg.lookup_best("item.likert.same.k5", "straightlined_share",
                            {"scale_points": 7, "items_per_block": 5, "keying": "same"})
    assert seven.value < 0.5 * five.value                  # the measured 7-point gap
    # unknown scale length never satisfies a scale-scoped cell
    unk = reg.lookup_best("item.likert.same.k5", "straightlined_share", {"items_per_block": 5, "keying": "same"})
    assert unk is not None and ".sp7." not in unk.entry_id


def test_engine_file_matches_the_evidence_file_and_is_measured():
    with open(os.path.join(REG_DIR, "item_process_7pt_v1306.json"), encoding="utf-8") as fh:
        eng = json.load(fh)
    with open(EVID, encoding="utf-8") as fh:
        ev = {e["entry_id"]: e for e in json.load(fh)["entries"]}
    assert len(eng["entries"]) >= 12
    for e in eng["entries"]:
        m = re.fullmatch(r"item\.likert\.(same|mixed)\.k(\d+)\.sp7\.straightlined_share", e["entry_id"])
        assert m, e["entry_id"]
        twin = ev[f"evidence.likert7.{m.group(1)}.k{m.group(2)}.straightlined_share"]
        assert e["value"] == twin["value"] and e["applicability"]["scale_points"] == [7]
        assert e["provenance"]["consumed_by_engine"] is True and e["tier"] == "T0_MEASURED"
        assert reg.entries()[e["entry_id"]].tier == reg.MEASURED


@pytest.mark.parametrize("points,expect_sp7", [(5, False), (7, True), (9, False)])
def test_identical_answer_pass_uses_the_seven_point_cell_only_for_seven_points(points, expect_sp7):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    scales = [{"name": "DV", "variable_name": "DV", "num_items": 5, "scale_points": points,
               "scale_min": 1, "scale_max": points, "type": "likert", "reverse_items": [],
               "detected_from_qsf": False, "_validated": True}]
    eng = EnhancedSimulationEngine(
        study_title="Warm glow giving", study_description="Judgements of a charity", sample_size=400,
        conditions=["A", "B"], factors=[{"name": "F", "levels": ["A", "B"]}], scales=scales,
        additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[], seed=3)
    eng.llm_generator.disable_permanently("test")
    eng.generate()
    rows = [x for x in eng._item_realism_log if x.get("stage") == "identical_answers"]
    assert rows, "the pass should have run on a 5-item block"
    assert (".sp7." in rows[0]["entry_id"]) is expect_sp7
    if expect_sp7:
        assert rows[0]["target_share"] == pytest.approx(reg.entries()["item.likert.same.k5.sp7.straightlined_share"].value, abs=1e-4)


def test_kb_headline_unmoved_by_registry_store():
    cov = reg.coverage_summary()
    assert cov["by_status"]["measured"] == 0          # store entries are not KB rows
    assert cov["sourced_entries"] == 1
    assert cov["shrinkage_verified"] is False


def test_block_aggregates_recompute_from_committed_profiles(evidence):
    """Offline reproducibility: each whole-block entry is the mean over the
    contributing block profiles stored alongside it."""
    prof = evidence["block_profiles"]
    for sp in (5, 7):
        grp = [p for p in prof if p["scale_points"] == sp]
        by_id = {e["entry_id"]: e for e in evidence["entries"]}
        for metric in ("item_sd_fraction_of_span", "floor_rate", "ceiling_rate",
                       "endpoint_occupancy", "midpoint_rate", "mean_abs_skew"):
            e = by_id[f"evidence.likert{sp}.blocks.{metric}"]
            assert e["value"] == pytest.approx(np.mean([p[metric] for p in grp]), abs=1e-12)
            assert e["dispersion"] == pytest.approx(
                np.std([p[metric] for p in grp], ddof=1), abs=1e-12)
        for k in (3, 5, 8):
            vals = [p["windows"]["any"][str(k)]["alpha"] for p in grp
                    if "windows" in p and str(k) in p["windows"]["any"]]
            e = by_id[f"evidence.likert{sp}.k{k}.aligned_alpha"]
            assert e["value"] == pytest.approx(np.mean(vals), abs=1e-12)


def test_reported_values_are_in_plausible_ranges(evidence):
    v = {e["entry_id"]: e["value"] for e in evidence["entries"]}
    for sp in (5, 7):
        assert 0.2 < v[f"evidence.likert{sp}.blocks.item_sd_fraction_of_span"] < 0.4
        assert 0.0 < v[f"evidence.likert{sp}.blocks.endpoint_occupancy"] < 0.7
        # reliability rises with block width (Spearman-Brown) in real data
        assert (v[f"evidence.likert{sp}.k3.aligned_alpha"]
                < v[f"evidence.likert{sp}.k5.aligned_alpha"]
                < v[f"evidence.likert{sp}.k8.aligned_alpha"])
        # straight-lining falls with width, and same-keyed far exceeds mixed-keyed
        assert v[f"evidence.likert{sp}.same.k3.straightlined_share"] > v[f"evidence.likert{sp}.same.k8.straightlined_share"]
        assert v[f"evidence.likert{sp}.same.k5.straightlined_share"] > 3 * v[f"evidence.likert{sp}.mixed.k5.straightlined_share"]
    # the measured finding that motivates scale-specific rates: 7-point same-keyed
    # blocks straight-line far less often than 5-point ones
    assert v["evidence.likert7.same.k5.straightlined_share"] < 0.5 * v["evidence.likert5.same.k5.straightlined_share"]


def test_ultimatum_acceptance_falls_with_lower_offers(evidence):
    v = {e["entry_id"]: e["value"] for e in evidence["entries"]}
    shares = [50, 45, 40, 25, 20, 15, 10, 5]
    rates = [v[f"evidence.game.ug_responder.accept_at_share{s:02d}"] for s in shares]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] == pytest.approx(1.0) and rates[-1] < 0.25
    e = {x["entry_id"]: x for x in evidence["entries"]}
    assert e["evidence.game.ug_responder.accept_at_share50"]["n_participants"] == 111


def test_binary_dictator_generosity_is_higher_when_ahead(evidence):
    v = {e["entry_id"]: e["value"] for e in evidence["entries"]}
    base = "evidence.game.binary_dictator.prosocial_choice_share."
    assert v[base + "dictator_ahead_in_both"] > 2 * v[base + "dictator_behind_in_both"]
    assert v[base + "cost_below_median"] > v[base + "cost_above_median"]


def test_game_and_effect_entries_state_their_width(evidence):
    for e in evidence["entries"]:
        if e["entry_id"].startswith("evidence.effect.") or e["entry_id"].endswith("public_minus_anonymous_d"):
            assert e["effect_scale"] == "d"
            assert e["dispersion_kind"] == "standard_error" and e["dispersion"] > 0
            assert any("single" in c.lower() or "one small" in c.lower() or "n=10" in c.lower()
                       for c in e["caveats"]), e["entry_id"]


def test_no_raw_data_or_names_committed(evidence):
    text = open(EVID, encoding="utf-8").read()
    assert len(text) < 2_000_000
    assert "Shrivastava" not in text and "Mahala" not in text      # participant names in a source file
    assert not any(f.endswith((".sav", ".zip")) for f in os.listdir(REG_DIR))


# --------------------------------------------------------------------------- tool helpers
def test_tool_empirical_key_and_windows_on_synthetic_data():
    t = _tool()
    rng = np.random.default_rng(7)
    n = 4000
    lat = rng.normal(size=n)
    cols = []
    for sign in (+1, +1, -1, +1, -1, -1):
        raw = sign * lat + rng.normal(scale=0.8, size=n)
        cols.append(np.clip(np.round(3 + raw), 1, 5))
    mat = np.column_stack(cols)
    key = t.empirical_key(mat)
    assert list(key) == [1, 1, -1, 1, -1, -1]
    w = t.window_stats(mat, key, widths=[3])
    assert w["mixed"][3]["alpha"] > 0.4          # alignment recovered the shared factor
    assert 3 not in w["same"] and w["mixed"][3]["windows"] == 4   # no all-same window here


def test_tool_cohens_d_matches_closed_form():
    t = _tool()
    a, b = np.array([3.0, 4, 5, 6]), np.array([1.0, 2, 3, 4])
    assert t.cohens_d(a, b) == pytest.approx(2.0 / np.sqrt(((3 * a.var(ddof=1)) + 3 * b.var(ddof=1)) / 6))


@pytest.mark.skipif(not os.environ.get("EVIDENCE_DATA_ROOT"),
                    reason="raw datasets are not committed; set EVIDENCE_DATA_ROOT to re-derive")
def test_rerun_reproduces_committed_file(tmp_path):
    t = _tool()
    out = tmp_path / "again.json"
    assert t.main([os.environ["EVIDENCE_DATA_ROOT"], "-o", str(out)]) == 0
    assert out.read_text() == open(EVID, encoding="utf-8").read()


# --------------------------------------------------------------------------- corroboration
def test_corroboration_file_is_not_a_registry_store(corroboration):
    assert "entries" not in corroboration            # so entries() never ingests it
    assert not any(k.startswith("corroboration") for k in reg.entries())


def test_corroboration_items_are_well_formed(corroboration):
    ids = set()
    for it in corroboration["items"]:
        assert it["id"] not in ids
        ids.add(it["id"])
        assert it["verdict"] in {"corroborated", "contradicted", "inconclusive"}
        assert it["urls"] and all(u.startswith("https://") for u in it["urls"])
        assert 0 < len(it["search_summary_states"].split()) <= 25, it["id"]
        assert it["note"].strip()
        assert "tier" not in it and "verified" not in json.dumps(it).lower().replace("not verified", "")


def test_corroboration_counts_add_up(corroboration):
    from collections import Counter
    c = Counter(i["verdict"] for i in corroboration["items"])
    assert dict(c) == corroboration["counts"]
    assert sum(c.values()) == len(corroboration["items"]) >= 45
    assert c["corroborated"] > 0 and c["inconclusive"] > 0 and c["contradicted"] > 0


def test_nothing_is_promoted_to_verified_without_a_quote():
    reg.reload_entries()
    for ent in reg.entries().values():
        if ent.tier in (reg.VERIFIED, reg.CORRECTED):
            assert str(ent.provenance.get("quote", "")).strip(), ent.entry_id
    # and a verified claim with no quote is downgraded by the loader
    e = reg._parse_entry({"entry_id": "x.y", "quantity": "mean", "value": 1.0,
                          "effect_scale": "d", "tier": "verified", "provenance": {}})
    assert e.tier == reg.CITED_UNCHECKED


def test_changed_values_keep_their_old_value(corroboration):
    """Anything the evidence pass changed keeps the old value as <field>_was in the recall band."""
    changed = [i for i in corroboration["items"] if i["changed_value"]]
    assert {i["id"] for i in changed} >= {"shrinkage_overall", "loyalty_program_meta",
                                          "personalization_meta", "social_exclusion_ostracism_meta"}
    for i in changed:
        p = reg.PROVENANCE.get(i["recall_key"])
        assert p is not None and p.status in reg.RECALL_TIERS, i["id"]
        assert any(k.endswith("_was") for k in p.corrected), i["id"]
        assert p.doi == "" and p.url == "" and p.quote == ""


def test_shrinkage_factor_lowered_on_the_gathered_ratios_and_still_unverified():
    from utils.empirical_provenance_data import RECALLED_SHRINKAGE as RS
    assert reg.shrinkage_factor() == pytest.approx(0.45)
    assert RS["corrected"]["factor_was"] == 0.60
    assert reg.EffectPolicy().min_retained == pytest.approx(0.35)
    assert reg.shrinkage_verified() is False
    assert reg.PROVENANCE["policy:publication_bias"].status in reg.RECALL_TIERS
    # the new factor is the study-weighted centre of the figures the summaries gave
    ratios = {"osc2015": (0.50, 100), "camerer2018": (0.50, 21), "camerer2016": (0.66, 18),
              "manylabs2": (0.25, 28), "kvarven2020": (0.33, 15)}
    centre = sum(r * k for r, k in ratios.values()) / sum(k for _, k in ratios.values())
    assert 0.40 <= reg.shrinkage_factor() <= 0.50 and abs(centre - 0.46) < 0.01


def test_relabelled_entries_no_longer_claim_to_be_meta_analyses():
    from utils import scientific_knowledge_base as kb
    for key in ("loyalty_program_meta", "personalization_meta"):
        e = kb.META_ANALYTIC_DB[key]
        assert e.source.startswith("NOT a meta-analysis") and e.n_studies == 0 and e.n_participants == 0
        assert "judgement" in e.notes
        assert reg.provenance_of("meta", key).corrected["n_studies_was"] in (35, 40)
    assert kb.META_ANALYTIC_DB["loyalty_program_meta"].effect_d == 0.20       # magnitude untouched
    assert kb.META_ANALYTIC_DB["personalization_meta"].effect_d == 0.32
    ost = kb.META_ANALYTIC_DB["social_exclusion_ostracism_meta"]
    assert ost.effect_d == -0.45 and "NOT that figure" in ost.notes
    assert "1.4" in reg.provenance_of("meta", "social_exclusion_ostracism_meta").note


def test_published_replication_ratios_are_recorded(corroboration):
    by = {i["id"]: i for i in corroboration["items"]}
    assert by["manylabs2_ratio"]["verdict"] == "contradicted" and "0.25" in by["manylabs2_ratio"]["note"]
    assert by["shrinkage_overall"]["changed_value"] is True
    assert all(by[k]["resolution"] for k in ("manylabs2_ratio", "jm_trust_k", "social_exclusion_ostracism_meta",
                                             "loyalty_program_meta", "personalization_meta"))
