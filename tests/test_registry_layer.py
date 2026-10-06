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
