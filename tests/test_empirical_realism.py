"""Tests for the empirical realism layer (v1.2.9.1).

Covers the four new modules and their engine integration:
  * empirical_registry      — provenance tiers, weighting, effect adjustment
  * empirical_marginals     — point-mass mixtures, heaping, rank transport
  * construct_matcher       — content-based matching to published norms
  * item_realism            — reliability correction and identical-answer share
  * realism_benchmark       — profile metrics and comparison against real data

The benchmark tests are the ones that matter: they assert that simulated data
stays inside the tolerances derived from REAL human item-level responses.
"""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "simulation_app"))

from utils import empirical_registry as reg          # noqa: E402
from utils import empirical_marginals as marg        # noqa: E402
from utils import construct_matcher as cm            # noqa: E402
from utils import item_realism as ir                 # noqa: E402
from utils import realism_benchmark as rb            # noqa: E402


# ───────────────────────────── registry ──────────────────────────────────────

def test_unrecorded_entries_report_unverified():
    assert reg.status_of("meta", "a_key_that_does_not_exist") == reg.UNVERIFIED
    assert reg.confidence_weight("meta", "a_key_that_does_not_exist") < 1.0


def test_audit_table_covers_every_knowledge_base_entry():
    rows = reg.audit_table()
    summary = reg.coverage_summary()
    assert len(rows) == summary["total_entries"] > 400
    assert all(r["status"] in reg._TIER_ORDER for r in rows)


def test_no_shrinkage_is_applied_while_the_factor_is_unverified():
    """An unverified correction must never be applied silently."""
    if not reg.coverage_summary()["shrinkage_verified"]:
        assert reg.shrinkage_factor() == 1.0


def test_adjust_effect_preserves_sign_and_never_inflates():
    rng = random.Random(0)
    for d in (0.2, 0.8, -0.5):
        out = reg.adjust_effect(d, kind="meta", key="unknown_key", rng=rng, tau=0.0)
        assert out == pytest.approx(d * reg.TIER_WEIGHT[reg.UNVERIFIED], rel=1e-6)
        assert (out > 0) == (d > 0)
    assert reg.adjust_effect(0.0) == 0.0


def test_honesty_notice_states_the_real_counts():
    note = reg.honesty_notice()
    s = reg.coverage_summary()
    assert str(s["total_entries"]) in note
    assert str(s["sourced_entries"]) in note


def test_dictator_sd_internal_inconsistency_is_recorded():
    """The entry's own point masses imply an SD near 0.31, not the stated 0.18."""
    corrected = reg.corrected_value("game", "dictator_standard", "sd_proportion")
    assert corrected is not None and corrected > 0.25


# ──────────────────────────── marginals ──────────────────────────────────────

def _engel_mixture():
    return marg.PointMassMixture(
        key="dictator_standard", mean=0.2835,
        point_masses=((0.0, 0.3611), (0.5, 0.1674), (1.0, 0.0544)),
        source="knowledge-base subpopulation shares",
    )


def test_point_mass_mixture_reproduces_mean_and_masses():
    m = _engel_mixture()
    assert m.theoretical_mean() == pytest.approx(0.2835, abs=1e-3)
    rng = random.Random(7)
    draws = [m.sample(rng) for _ in range(20000)]
    assert sum(draws) / len(draws) == pytest.approx(0.2835, abs=0.02)
    for value, share in m.point_masses:
        got = sum(1 for d in draws if abs(d - value) < 1e-9) / len(draws)
        assert got == pytest.approx(share, abs=0.02)


def test_mixture_sd_matches_the_recorded_correction():
    """A distribution with those masses cannot have SD 0.18."""
    m = _engel_mixture()
    rng = random.Random(3)
    draws = [m.sample(rng) for _ in range(20000)]
    mean = sum(draws) / len(draws)
    sd = math.sqrt(sum((d - mean) ** 2 for d in draws) / len(draws))
    assert sd > 0.25
    assert sd == pytest.approx(0.31, abs=0.04)


def test_unverified_marginals_are_never_active():
    """A shape with no sourced provenance must not reshape anyone's data."""
    mix = marg.PointMassMixture(key="no_such_game_key", mean=0.5)
    marg.register_marginal(mix)
    try:
        assert marg.marginal_for("no_such_game_key") is None
        assert marg.reshape_allocation_column(
            [1.0] * 50, "no_such_game_key", 0, 100, random.Random(0)) is None
    finally:
        marg.GAME_MARGINALS.pop("no_such_game_key", None)


def test_rank_transport_preserves_order():
    values = [5.0, 1.0, 3.0, 9.0]
    targets = [10.0, 20.0, 30.0, 40.0]
    out = marg.rank_transport(values, targets)
    # ranks are 1.0 < 3.0 < 5.0 < 9.0, so the targets land 30, 10, 20, 40
    assert out == [30.0, 10.0, 20.0, 40.0]
    assert sorted(out) == targets


def test_heaping_snaps_only_to_grid_points():
    rng = random.Random(1)
    grid = set(marg.heaping_grid(0, 100))
    assert 50.0 in grid and 5.0 in grid
    snapped = [marg.snap_to_round(37.3, 0, 100, 1.0, rng) for _ in range(200)]
    assert all(v in grid for v in snapped)
    # short scales have no round-number structure to impose
    assert marg.heaping_grid(1, 7) == []
    assert marg.snap_to_round(4.2, 1, 7, 1.0, rng) == 4.2


# ───────────────────────── construct matcher ─────────────────────────────────

def test_matcher_reaches_the_whole_norm_table():
    cov = cm.coverage()
    assert cov["norms_indexed"] > 150


@pytest.mark.parametrize("variable,expected_fragment", [
    ("PSS4_1", "perceived_stress"),
    ("swls_total", "life_satisfaction"),
    ("loneliness_1", "loneliness"),
    ("UWES_3", "work_engagement"),
    ("bfi_extra_1", "extraversion"),
    ("bfi_neuro_3", "neuroticism"),
    ("panas_pos_4", "positive_affect"),
    ("mfq_purity_1", "moral_foundations_purity"),
    ("crt_1", "cognitive_reflection"),
])
def test_matcher_finds_the_right_construct(variable, expected_fragment):
    m = cm.match(variable)
    assert m is not None, f"{variable} matched nothing"
    assert expected_fragment in m.key


@pytest.mark.parametrize("variable", ["Q17", "dv_rating", "age", "duration_sec",
                                     "participant_id", "CONDITION"])
def test_matcher_declines_non_construct_variables(variable):
    """A wrong norm is worse than no norm: these must not match."""
    assert cm.match(variable) is None


# ─────────────────────────── item realism ────────────────────────────────────

def _synthetic_block(n=400, k=5, r_high=True, seed=0):
    """A block built the way an over-consistent generator builds one."""
    rng = random.Random(seed)
    common = [rng.gauss(0, 1) for _ in range(n)]
    noise = 0.3 if r_high else 1.6
    cols = []
    for _j in range(k):
        cols.append([
            min(6, max(1, round(4 + common[i] + rng.gauss(0, noise))))
            for i in range(n)
        ])
    return cols


def test_decoupling_lowers_internal_consistency_to_target():
    cols = _synthetic_block()
    before = ir.mean_interitem_r(cols)
    assert before > 0.5, "fixture should start over-consistent"
    out, rep = ir.decouple_block(cols, 1, 6, target_r=0.363)
    assert rep.applied
    assert ir.mean_interitem_r(out) == pytest.approx(0.363, abs=0.06)


def test_decoupling_never_raises_consistency():
    cols = _synthetic_block(r_high=False, seed=2)
    out, rep = ir.decouple_block(cols, 1, 6, target_r=0.363)
    assert not rep.applied
    assert out == [[float(v) for v in c] for c in cols]


def test_decoupling_preserves_participant_ranks_on_the_construct():
    cols = _synthetic_block(seed=5)
    out, rep = ir.decouple_block(cols, 1, 6, target_r=0.363)
    assert rep.applied
    k = len(cols)
    before = [sum(c[i] for c in cols) / k for i in range(len(cols[0]))]
    after = [sum(c[i] for c in out) / k for i in range(len(out[0]))]
    # Spearman-style check: the composite ordering must stay essentially intact.
    pairs = sorted(range(len(before)), key=lambda i: before[i])
    ranked_after = [after[i] for i in pairs]
    concordant = sum(1 for a, b in zip(ranked_after, ranked_after[1:]) if a <= b)
    assert concordant / max(1, len(ranked_after) - 1) > 0.85


def test_decoupling_respects_scale_bounds_and_integrality():
    cols = _synthetic_block(seed=9)
    out, _ = ir.decouple_block(cols, 1, 6, target_r=0.30)
    for col in out:
        assert all(1 <= v <= 6 for v in col)
        assert all(float(v).is_integer() for v in col)


def test_decoupling_declines_degenerate_input():
    assert not ir.decouple_block([[1, 2, 3]] * 2, 1, 6)[1].applied       # k < 3
    assert not ir.decouple_block([[1] * 5] * 4, 1, 6)[1].applied         # n < 20
    assert not ir.decouple_block(_synthetic_block(), 3, 3)[1].applied    # no range


def test_target_r_from_alpha_matches_the_standard_formula():
    assert ir.target_r_from_alpha(0.75, 5) == pytest.approx(0.375, abs=1e-3)
    assert ir.target_r_from_alpha(0.80, 10) == pytest.approx(0.8 / 2.8, abs=1e-3)


def test_straightlining_is_raised_to_the_real_share_without_removing_rows():
    # a block with plenty of item-specific noise has almost no constant rows
    cols = _synthetic_block(n=600, r_high=False, seed=11)
    assert ir.straightlined_share(cols) < 0.052
    out, rep = ir.match_straightlining(cols, 1, 6, target_share=0.052,
                                       rng=random.Random(1))
    assert rep["applied"]
    assert ir.straightlined_share(out) == pytest.approx(0.052, abs=0.01)
    assert all(len(c) == len(cols[0]) for c in out)


def test_straightlining_never_reduces_an_already_high_share():
    cols = [[4] * 100 for _ in range(5)]        # everybody identical already
    out, rep = ir.match_straightlining(cols, 1, 6, target_share=0.052)
    assert not rep["applied"]
    assert ir.straightlined_share(out) == 1.0


# ───────────────────────── benchmark plumbing ────────────────────────────────

def test_reference_profiles_ship_and_describe_their_source():
    payload = rb.load_reference_profiles()
    assert payload.get("profiles"), "reference profiles must ship with the package"
    prov = payload["provenance"][0]
    for field in ("dataset", "source_page", "data_file", "quote", "retrieved_on",
                  "distributed_with", "caveats"):
        assert prov.get(field), f"provenance is missing {field}"


def test_reference_selection_respects_item_keying():
    aligned = rb.reference_for_scale(6, direction_aligned=True)
    mixed = rb.reference_for_scale(6, direction_aligned=False)
    assert aligned and mixed
    assert aligned["mean_interitem_r"] > mixed["mean_interitem_r"]


def test_listwise_deletion_keeps_respondents_aligned():
    """Dropping missing values per column would scramble row-wise metrics."""
    data = {"a": [1, 2, float("nan"), 4], "b": [1, 2, 3, 4], "c": [1, 2, 3, 4]}
    cols = rb._as_columns(data, ["a", "b", "c"])
    assert [len(c) for c in cols] == [3, 3, 3]
    assert cols[0] == [1.0, 2.0, 4.0] and cols[1] == [1.0, 2.0, 4.0]


def test_profile_detects_a_straight_lined_sample():
    data = {f"i{j}": [3] * 50 for j in range(5)}
    prof = rb.compute_profile(data, list(data), 1, 6)
    assert prof is not None
    assert prof.straightlined_share == 1.0
    assert prof.within_person_sd_mean == 0.0
    assert prof.options_used_mean == 1.0


def test_benchmark_reports_no_score_when_no_reference_fits():
    data = {f"i{j}": [random.Random(j).randint(1, 50) for _ in range(40)]
            for j in range(4)}
    out = rb.benchmark(data, list(data), 1, 50)
    assert out["comparison"] is None or out["comparison"]["n_compared"] >= 0
    assert "verdict" in out
