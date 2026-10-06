"""Reproducibility and missing-data crash regressions (v1.2.8.8).

Same seed + same design + same settings must give a byte-identical dataset, and
generation must not crash when item-level missingness / dropout is enabled.
"""
import hashlib

import pytest

from utils.enhanced_simulation_engine import EnhancedSimulationEngine


def _engine(seed, n=60, missing=0.0, dropout=0.0, oe=True):
    return EnhancedSimulationEngine(
        study_title="Trust in AI advisors",
        study_description="AI label effect on trust",
        sample_size=n,
        conditions=["Control", "Treatment"],
        factors=[],
        scales=[{"name": "Trust", "items": 5, "scale_points": 7}],
        additional_vars=[],
        demographics={"age": True, "gender": True},
        open_ended_questions=(
            [{"name": "Why", "text": "Why did you rate trust this way?"}] if oe else []
        ),
        study_context={},
        seed=seed,
        missing_data_rate=missing,
        dropout_rate=dropout,
    )


def _digest(df) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def test_same_seed_is_byte_identical_including_text_and_missingness():
    a, _ = _engine(5, missing=0.03, dropout=0.05).generate()
    b, _ = _engine(5, missing=0.03, dropout=0.05).generate()
    assert _digest(a) == _digest(b)


def test_different_seed_changes_the_data():
    a, _ = _engine(5).generate()
    c, _ = _engine(6).generate()
    assert _digest(a) != _digest(c)


def test_run_id_has_no_wall_clock_component():
    e1, e2 = _engine(11), _engine(11)
    assert e1.run_id == e2.run_id
    assert "S0000000011" in e1.run_id


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_missing_data_does_not_crash_consistency_audit(seed):
    df, _ = _engine(seed, n=80, missing=0.12, dropout=0.08, oe=False).generate()
    assert len(df) == 80
