"""Tests for the applicability-guarded registry, design signatures and the run ledger.

The point of these is narrow: a benchmark must fire exactly where it applies and
decline everywhere else. A registry that returns a plausible wrong number is worse
than one that returns nothing, because the caller's fallback is explicit and
inspectable while a wrong number is not.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

from utils import empirical_registry as R                      # noqa: E402
from utils.design_signature import DesignSignature, for_block, infer_keying  # noqa: E402


# --------------------------------------------------------------------------
# Entries load, and every one of them is well formed
# --------------------------------------------------------------------------

def test_registry_loads_entries():
    assert len(R.entries()) > 0


def test_every_entry_declares_a_known_effect_scale():
    for ent in R.entries().values():
        assert ent.effect_scale in R.EFFECT_SCALES, ent.entry_id


def test_measured_entries_carry_a_reproducible_derivation():
    """T0 means the evidence is the data itself, so the script and the files must
    be named or the tier is downgraded at load time."""
    for ent in R.entries().values():
        if ent.tier == R.MEASURED:
            assert ent.provenance.get("script"), ent.entry_id
            assert ent.provenance.get("sources"), ent.entry_id
            for src in ent.provenance["sources"]:
                assert len(src.get("sha256", "")) == 64, ent.entry_id


def test_only_evidenced_tiers_may_set_a_magnitude():
    for ent in R.entries().values():
        assert ent.may_set_magnitude == (ent.tier in (R.MEASURED, R.VERIFIED, R.CORRECTED))


# --------------------------------------------------------------------------
# Applicability guards
# --------------------------------------------------------------------------

def _sig(**kw):
    base = dict(scale_min=1, scale_max=5, n_items=10,
                reverse_flags=[0, 1] + [0] * 8, design_type="between")
    base.update(kw)
    return for_block(**base)


def test_five_point_benchmark_fires_on_a_five_point_block():
    hit = R.lookup_best("item.likert5", "endpoint_occupancy", _sig())
    assert hit is not None and hit.tier == R.MEASURED
    assert 0.0 < hit.value < 1.0


def test_five_point_benchmark_declines_on_a_seven_point_block():
    assert R.lookup_best("item.likert5", "endpoint_occupancy",
                         _sig(scale_max=7)) is None


def test_block_benchmark_declines_on_a_block_that_is_too_short():
    assert R.lookup_best("item.likert5", "item_sd", _sig(n_items=3)) is None


def test_keying_selects_a_different_straightlining_entry():
    """Straight-lining is a property of the instrument's keying. Same-keyed blocks
    straight-line an order of magnitude more often, so the two must not share an
    entry."""
    mixed = R.lookup_best("item.likert5", "straightlined_share", _sig())
    same = R.lookup_best("item.likert5", "straightlined_share",
                         _sig(reverse_flags=[0] * 10))
    assert mixed is not None and same is not None
    assert mixed.entry_id != same.entry_id
    assert same.value > 5 * mixed.value


def test_unknown_dimension_never_satisfies_a_stated_condition():
    app = R.Applicability(keying="mixed")
    assert not app.accepts(DesignSignature(keying="unknown"))
    assert app.accepts(DesignSignature(keying="mixed"))


def test_narrower_entry_wins_a_tie():
    narrow = R.Applicability(scale_points=(5,), items_per_block=(8, 10), keying="mixed")
    wide = R.Applicability(scale_points=(5,))
    assert narrow.specificity() > wide.specificity()


def test_lookup_requires_magnitude_permission_by_default():
    ent = next(iter(R.entries().values()))
    assert R.lookup(ent.entry_id, require_magnitude=False) is not None


# --------------------------------------------------------------------------
# Design signature
# --------------------------------------------------------------------------

def test_scale_points_derived_from_endpoints():
    assert for_block(scale_min=1, scale_max=7).scale_points == 7
    assert for_block(scale_min=0, scale_max=10).scale_points == 11


def test_non_integer_or_inverted_range_yields_no_scale_points():
    assert for_block(scale_min=1, scale_max=5.5).scale_points is None
    assert for_block(scale_min=7, scale_max=1).scale_points is None


def test_factorial_is_a_between_subjects_layout():
    assert for_block(design_type="factorial").design == "between"


def test_infer_keying():
    assert infer_keying(None) == "unknown"
    assert infer_keying([]) == "unknown"
    assert infer_keying([0, 0, 0]) == "same"
    assert infer_keying([0, "R", 0]) == "mixed"
    assert infer_keying([False, True]) == "mixed"


def test_position_fraction():
    s = for_block(scale_min=1, scale_max=5).with_position(50, 100)
    assert s.position_fraction == pytest.approx(0.5)
    assert for_block(scale_min=1, scale_max=5).position_fraction is None


def test_signature_key_is_stable():
    a = for_block(scale_min=1, scale_max=5, n_items=5)
    b = for_block(scale_min=1, scale_max=5, n_items=5)
    assert a.key() == b.key()


# --------------------------------------------------------------------------
# Shrinkage tier gate — the constant that would touch every effect in the system
# --------------------------------------------------------------------------

def test_shrinkage_refuses_unverified_provenance():
    with pytest.raises(R.ProvenanceTooWeak):
        R.set_shrinkage(0.6, 0.2, R.Provenance(status=R.UNVERIFIED))


def test_shrinkage_refuses_a_verified_record_with_no_quote():
    with pytest.raises(R.ProvenanceTooWeak):
        R.set_shrinkage(0.6, 0.2, R.Provenance(status=R.VERIFIED, doi="10.x/y"))


def test_shrinkage_refuses_tau_on_another_scale():
    prov = R.Provenance(status=R.VERIFIED, quote="the replication estimate was ...")
    with pytest.raises(R.ProvenanceTooWeak):
        R.set_shrinkage(0.6, 0.2, prov, tau_scale="fisher_z")


def test_no_shrinkage_is_installed_by_default():
    """Nothing has been verified, so nothing is corrected. The honest default."""
    assert R.shrinkage_factor() == 1.0
    assert R.default_tau() == 0.0


# --------------------------------------------------------------------------
# Run ledger
# --------------------------------------------------------------------------

def test_ledger_reports_the_mix_for_this_run():
    led = R.RunLedger()
    assert led.notice() == ""
    hit = R.lookup_best("item.likert5", "item_sd", _sig())
    led.record_lookup(hit, "Trust block")
    led.record("meta:anchoring", R.UNVERIFIED, 0.4, "CONDITION")
    s = led.summary()
    assert s["applied"] == 2 and s["evidenced"] == 1 and s["asserted"] == 1
    assert "1 of 2" in led.notice()


# --------------------------------------------------------------------------
# Marginal shape: maximum-entropy targets and rank transport
# --------------------------------------------------------------------------

import math                                                    # noqa: E402
import random                                                  # noqa: E402

from utils.item_realism import maxent_discrete, match_item_dispersion  # noqa: E402


def _moments(probs, lo):
    ks = list(range(lo, lo + len(probs)))
    mu = sum(p * k for p, k in zip(probs, ks))
    var = sum(p * (k - mu) ** 2 for p, k in zip(probs, ks))
    return mu, math.sqrt(var)


@pytest.mark.parametrize("lo,hi,mean,sd", [
    (1, 5, 3.0, 1.18), (1, 7, 4.2, 1.77), (1, 9, 4.9, 2.34), (1, 5, 2.0, 1.00),
])
def test_maxent_recovers_the_requested_moments(lo, hi, mean, sd):
    mu, s = _moments(maxent_discrete(lo, hi, mean, sd), lo)
    assert mu == pytest.approx(mean, abs=0.01)
    assert s == pytest.approx(sd, abs=0.01)


def test_maxent_is_flatter_than_a_discretised_normal():
    """Real 5-point items have negative excess kurtosis and sit on the endpoints;
    a peaked distribution is the tell this pass exists to remove."""
    p = maxent_discrete(1, 5, 3.0, 1.18)
    assert p[0] + p[-1] > 0.15
    assert max(p) < 0.40


def _block(gap, n=400, k=5, within=0.6, seed=1, blocked=True):
    rng = random.Random(seed)
    cond = (["A"] * (n // 2) + ["B"] * (n // 2)) if blocked else (["A", "B"] * (n // 2))
    cols = [[min(5, max(1, round(rng.gauss(3.0 + (gap if cond[i] == "B" else 0.0), within))))
             for i in range(n)] for _ in range(k)]
    return cols, cond


def _composite_d(cols, cond):
    n = len(cols[0])
    comp = [sum(c[i] for c in cols) / len(cols) for i in range(n)]
    a = [comp[i] for i in range(n) if cond[i] == "A"]
    b = [comp[i] for i in range(n) if cond[i] == "B"]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((v - ma) ** 2 for v in a) / (len(a) - 1)
    vb = sum((v - mb) ** 2 for v in b) / (len(b) - 1)
    return (mb - ma) / math.sqrt((va + vb) / 2)


def test_dispersion_match_hits_the_measured_target():
    cols, cond = _block(0.3)
    new, rep = match_item_dispersion(cols, 1, 5, condition_labels=cond,
                                     rng=random.Random(7))
    assert rep.sd_after > rep.sd_before
    assert rep.sd_after == pytest.approx(0.295 * 4, abs=0.15)
    assert rep.endpoint_after > rep.endpoint_before


def test_dispersion_match_does_not_manufacture_an_effect():
    """Regression. Breaking rank ties by row index orders tied respondents by their
    position in the frame, which when the frame is built condition by condition IS
    the condition: it turned a null design (d = 0.06) into d = 1.65. Ties must be
    broken by noise independent of the design."""
    cols, cond = _block(0.0, blocked=True)
    before = _composite_d(cols, cond)
    new, _ = match_item_dispersion(cols, 1, 5, condition_labels=cond,
                                   rng=random.Random(7))
    after = _composite_d(new, cond)
    assert abs(before) < 0.2
    assert abs(after) < 0.2


def test_dispersion_match_preserves_a_real_effect():
    for gap in (0.15, 0.3, 0.6):
        cols, cond = _block(gap)
        before = _composite_d(cols, cond)
        new, _ = match_item_dispersion(cols, 1, 5, condition_labels=cond,
                                       rng=random.Random(7))
        after = _composite_d(new, cond)
        assert after > 0.5 * before, f"gap={gap}: {before:.3f} -> {after:.3f}"


def test_dispersion_match_declines_on_degenerate_input():
    cols, cond = _block(0.3, n=10)
    _, rep = match_item_dispersion(cols, 1, 5, condition_labels=cond)
    assert rep.skipped and rep.adjusted_items == 0
    cols, cond = _block(0.3)
    cols[0][0] = float("nan")
    _, rep = match_item_dispersion(cols, 1, 5, condition_labels=cond)
    assert rep.skipped == "missing values present"


def test_dispersion_match_returns_integers_on_an_integer_scale():
    cols, cond = _block(0.3)
    new, _ = match_item_dispersion(cols, 1, 5, condition_labels=cond,
                                   rng=random.Random(7))
    for col in new:
        assert all(isinstance(v, int) and 1 <= v <= 5 for v in col)


# --------------------------------------------------------------------------
# Straight-lining: width- and keying-conditional, and bidirectional
# --------------------------------------------------------------------------

from utils.item_realism import match_straightlining, straightlined_share  # noqa: E402


def test_straightlining_entries_fall_with_block_width():
    """Straight-lining a 3-item block is far easier than a 10-item one. A single
    global share is therefore wrong by an order of magnitude at one end or other."""
    vals = {}
    for k in range(3, 11):
        ent = R.entries().get(f"item.likert.mixed.k{k}.straightlined_share")
        if ent is not None:
            vals[k] = ent.value
    assert len(vals) >= 6
    ks = sorted(vals)
    assert all(vals[a] > vals[b] for a, b in zip(ks, ks[1:]))
    assert vals[ks[0]] > 10 * vals[ks[-1]]


def test_same_keyed_blocks_straightline_more_than_mixed_keyed():
    for k in (3, 5, 8):
        same = R.entries().get(f"item.likert.same.k{k}.straightlined_share")
        mixed = R.entries().get(f"item.likert.mixed.k{k}.straightlined_share")
        assert same is not None and mixed is not None
        assert same.value > 2 * mixed.value


def test_single_block_entries_say_so():
    for ent in R.entries().values():
        n = len(ent.provenance.get("contributing_blocks") or [])
        if n == 1:
            assert any("single contributing block" in c for c in ent.caveats), ent.entry_id


def _sl_fixture(share, n=600, k=8, seed=5):
    rng = random.Random(seed)
    cols = [[0.0] * n for _ in range(k)]
    n_const = int(share * n)
    for i in range(n):
        row = ([rng.choice([1, 7])] * k if i < n_const
               else [rng.randint(1, 7) for _ in range(k)])
        for j in range(k):
            cols[j][i] = float(row[j])
    return cols


def test_straightlining_raises_a_share_that_is_too_low():
    cols = _sl_fixture(0.002)
    new, rep = match_straightlining(cols, 1, 7, target_share=0.036,
                                    rng=random.Random(1))
    assert rep["applied"]
    assert straightlined_share(new) == pytest.approx(0.036, abs=0.005)


def test_straightlining_lowers_a_share_that_is_too_high():
    """New in v1.2.9.3. The marginal-shape pass produces constant rows readily, so
    the pass has to be able to correct downward too — an 8-item block was coming
    out at 3.7% where the mixed-keyed real rate is 0.30%."""
    cols = _sl_fixture(0.10)
    new, rep = match_straightlining(cols, 1, 7, target_share=0.003,
                                    rng=random.Random(1))
    assert rep["applied"] and rep.get("n_broken", 0) > 0
    assert straightlined_share(new) == pytest.approx(0.003, abs=0.003)


def test_breaking_a_row_stays_on_scale():
    cols = _sl_fixture(0.10)
    new, _ = match_straightlining(cols, 1, 7, target_share=0.003,
                                  rng=random.Random(1))
    assert all(1 <= v <= 7 for col in new for v in col)


def test_validator_straightlining_range_follows_block_width():
    from utils.hbs_validator import HBSValidator
    v = HBSValidator()
    lo3, hi3 = v._straightlining_range(3)
    lo10, hi10 = v._straightlining_range(10)
    assert hi3 > hi10 and lo3 > lo10
    # The old flat benchmark forced a 10-item block to 3-8%; measured is far lower.
    assert hi10 < 0.03


# ---------------------------------------------------------------------------
# Literature fallback: the reference arm, and reproducibility
# ---------------------------------------------------------------------------

def _dissonance_engine(seed=11):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    return EnhancedSimulationEngine(
        study_title="Dissonance", study_description="", sample_size=20,
        conditions=["cognitive_dissonance_control", "cognitive_dissonance_induced"],
        factors=[],
        scales=[{
            "name": "attitude", "variable_name": "attitude", "num_items": 4,
            "scale_points": 7, "_validated": True, "scale_min": 1, "scale_max": 7,
            "question_text": "How favourable is your attitude toward the essay topic?",
            "dv_description": "attitude",
        }],
        additional_vars=[], demographics={}, seed=seed,
    )


def test_literature_fallback_leaves_the_control_arm_at_zero():
    """A control label repeats the paradigm it controls for.

    Matched on content, `cognitive_dissonance_control` shares every word with
    `cognitive_dissonance_induced`, so without a guard both arms take the same
    published effect and the contrast vanishes -- the fallback meant to rescue a
    null design would recreate one.
    """
    engine = _dissonance_engine()
    control = engine._get_effect_for_condition("cognitive_dissonance_control", "attitude")
    treated = engine._get_effect_for_condition("cognitive_dissonance_induced", "attitude")
    # Not exactly zero: the keyword pipeline's stable-hash jitter leaves noise at
    # d ~ 0.0001. What matters is that no published effect was applied to it.
    assert abs(control) < 0.01
    assert abs(treated - control) > 0.01


def test_stable_rng_depends_only_on_the_seed():
    """Two engines built with the same seed must draw the same numbers."""
    a = _dissonance_engine(11)._stable_rng("literature-effect", "cond", "dv").random()
    b = _dissonance_engine(11)._stable_rng("literature-effect", "cond", "dv").random()
    c = _dissonance_engine(12)._stable_rng("literature-effect", "cond", "dv").random()
    assert a == b
    assert a != c


def test_survey_wording_reaches_a_numbered_item():
    """`attitude_3` must resolve back to its scale's question text."""
    q, item = _dissonance_engine()._survey_wording_for("attitude_3")
    assert "attitude" in q.lower()
    assert item == "attitude"


# ---------------------------------------------------------------------------
# Long blocks: the straight-lining target must not fall back to a wrong constant
# ---------------------------------------------------------------------------

def test_long_instruments_use_the_full_instrument_rate_not_the_default():
    """A 22-50 item block has a measured rate of ~0.12%, not the 5.2% default.

    The per-width entries stop at k=10. Falling back to the global default past
    that would put roughly 40x too many respondents on a constant row, and since
    the pass corrects downward too it would create them rather than merely fail
    to remove them.
    """
    from utils import design_signature, empirical_registry

    sig = design_signature.for_block(scale_min=1, scale_max=5, n_items=22,
                                     keying="mixed")
    width = empirical_registry.lookup_best(
        "item.likert.mixed.k22", "straightlined_share", sig)
    full = empirical_registry.lookup_best(
        "item.likert5.full_instrument", "straightlined_share", sig)

    assert width is None, "there is no measured 22-item width entry to find"
    assert full is not None
    assert full.value < 0.01, "a long instrument's rate is well under one percent"


def test_an_unmeasured_block_width_declines_rather_than_guessing():
    """Between the per-width entries and the full-instrument one, nothing applies.

    12 items is past the measured widths and short of the full-instrument range,
    so both lookups decline and the engine leaves the block untouched.
    """
    from utils import design_signature, empirical_registry

    sig = design_signature.for_block(scale_min=1, scale_max=5, n_items=12,
                                     keying="mixed")
    assert empirical_registry.lookup_best(
        "item.likert.mixed.k12", "straightlined_share", sig) is None
    assert empirical_registry.lookup_best(
        "item.likert5.full_instrument", "straightlined_share", sig) is None


# ---------------------------------------------------------------------------
# Acronyms that map to several subscales
# ---------------------------------------------------------------------------

def test_an_ambiguous_acronym_declines_instead_of_picking_a_subscale():
    """Every `BFI_*` item used to come back as agreeableness, wording ignored.

    The candidates all share the acronym, so a ranking built on it orders them by
    nothing; returning the first applied agreeableness's published mean and shape
    to a neuroticism item. No distinguishing word means no match.
    """
    from utils.construct_matcher import match

    assert match(variable_name="BFI_4",
                 question_text="I get nervous easily") is None
    assert match(variable_name="BFI_1",
                 question_text="I am outgoing and sociable") is None


def test_an_unambiguous_acronym_still_matches():
    """The guard must not cost the cases that were already right."""
    from utils.construct_matcher import match

    pss = match(variable_name="PSS4_1")
    assert pss is not None and pss.key == "perceived_stress_pss"

    mbi = match(variable_name="MBI_3",
                question_text="I feel emotionally exhausted by my work")
    assert mbi is not None and mbi.key == "burnout_emotional_exhaustion"
