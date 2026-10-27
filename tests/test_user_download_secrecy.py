"""The user's download must not reveal how the data were simulated (v1.3.1.2).

The ZIP the user downloads and the package emailed to the owner used to be the same bytes, so
every download carried the seed, the persona roster, the trait averages, the configured and
observed Cohen's d per contrast (the answer key), the calibration/registry provenance and the
engine's identity. ``utils.user_package`` now derives the user ZIP from the owner package.

These tests fail if any of that comes back. The expensive one builds a real bundle end to end
(engine -> Qualtrics export -> codebook -> prep scripts -> study summary -> user package) and
scans every delivered file; the cheap ones pin the allowlists, the run-id scrubbing and the
wiring in ``app.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from utils.user_package import (
    OWNER_ONLY_FILE_NAMES,
    build_user_package,
    find_simulation_internals,
    public_run_id,
    sanitize_codebook,
    sanitize_diagnostics_csv,
    sanitize_metadata,
    sanitize_study_summary,
)

_APP_PY = Path(__file__).resolve().parent.parent / "simulation_app" / "app.py"

#: Metadata keys that carry simulation internals. None may appear in the user's metadata file.
_FORBIDDEN_METADATA_KEYS = (
    "seed",
    "personas_used",
    "persona_distribution",
    "persona_by_condition",
    "trait_averages_by_condition",
    "trait_averages_overall",
    "effect_sizes_configured",
    "effect_sizes_observed",
    "effect_sizes_applied",
    "calibration_provenance",
    "detected_domains",
    "abe3_engine",
    "generation_method",
    "generation_method_label",
    "llm_response_stats",
    "scale_generation_log",
    "cross_dv_correlation",
    "consistency_audit",
    "missing_data",
    "run_archive_path",
    "run_audit_summary",
    "run_improvement_log",
)

#: Diagnostics columns that must not be delivered.
_FORBIDDEN_DIAGNOSTIC_COLUMNS = (
    "RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED", "ABE3_ResponseStyle", "_Generation_Source",
)


# ──────────────────────────────────────────────────────────────────────────────
# The scanner itself
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Persona Distribution (Simulated Response Styles)",
    "| Careless_Responder | ~3 | Standard response patterns |",
    '"seed": 7',
    "Random Seed | 7",
    '"cohens_d": 0.5',
    "| Trust | Control | Treatment | 4.86 | 4.44 | 0.29 | Your specified effect(s) | +0.50 |",
    "Inferred from condition names",
    "calibration_provenance",
    "shrinkage_factor",
    "ABE3_ResponseStyle",
    "Adaptive Behavioral Engine 3.0",
    "Detected Domains: political_psychology",
    "base prevalence rates from survey methodology research (Krosnick, 1991)",
    "Census-weighted education level",
    "7-point party identification (ANES)",
    "acquiescence 0.5365",
    "## Simulation Intelligence Report",
    "cross_dv_correlation",
])
def test_scanner_catches_known_internals(text):
    assert find_simulation_internals(text), f"scanner missed an internal: {text!r}"


@pytest.mark.parametrize("text", [
    "Simulated_Data.csv",
    "The responses in this dataset are simulated, not collected from people.",
    "Max_Straight_Line: Maximum consecutive identical responses",
    "| Flag_StraightLine | Flagged for straight-lining: 1=Yes, 0=No |",
    "| Cohen's d | 0.20 | 0.50 | 0.80 | Two-group comparisons |",
    "Exclude_Recommended | Recommended for exclusion: 1=Yes, 0=No",
    "data <- read_csv('Simulated_Data.csv', show_col_types = FALSE)",
    "One-way ANOVA for comparing condition means",
])
def test_scanner_allows_legitimate_codebook_vocabulary(text):
    assert find_simulation_internals(text) == [], f"scanner is too broad: {text!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Allowlists
# ──────────────────────────────────────────────────────────────────────────────

def test_metadata_is_rebuilt_from_an_allowlist():
    full = {
        "run_id": "PILOT_S0000000007",
        "seed": 7,
        "study_title": "Trust study",
        "sample_size": 60,
        "conditions": ["Control", "Treatment"],
        "personas_used": ["careless_responder"],
        "persona_distribution": {"counts": {"careless_responder": 3}},
        "trait_averages_overall": {"acquiescence": 0.53},
        "effect_sizes_observed": [{"variable": "Trust", "cohens_d": 0.29}],
        "calibration_provenance": {"declare_first": True},
        "abe3_engine": {"enabled": True, "version": "3.0.0"},
        "detected_domains": ["political_psychology"],
        "generation_method_label": "Adaptive Behavioral Engine 3.0",
        "_future_internal_nobody_remembered": {"shrinkage_factor": 0.45},
        "scales": [{"name": "Trust", "num_items": 5, "scale_points": 7, "_validated": True,
                    "persona_weighting": "x"}],
        "column_descriptions": {
            "SIMULATION_SEED": "Random seed for reproducibility",
            "Mean_Item_RT_ms": "Mean response time per scale item in ms (Yan & Tourangeau, 2008)",
            "ABE3_PartyID": "7-point party identification (ANES)",
            "Trust_1": "Trust item 1 (1-7)",
        },
    }
    clean = sanitize_metadata(full)
    for key in _FORBIDDEN_METADATA_KEYS:
        assert key not in clean, f"{key} reached the user's metadata"
    assert "_future_internal_nobody_remembered" not in clean, "allowlist leaked an unknown key"
    assert clean["study_title"] == "Trust study"
    assert clean["run_id"] != "PILOT_S0000000007"
    assert clean["scales"] == [{"name": "Trust", "num_items": 5, "scale_points": 7}]
    assert "SIMULATION_SEED" not in clean["column_descriptions"]
    assert "ABE3_PartyID" not in clean["column_descriptions"]
    assert clean["column_descriptions"]["Trust_1"] == "Trust item 1 (1-7)"
    assert find_simulation_internals(json.dumps(clean)) == []


def test_diagnostics_columns_are_filtered_and_renamed():
    csv_text = (
        "ResponseId,PARTICIPANT_ID,RUN_ID,SIMULATION_MODE,SIMULATION_SEED,Trust_mean,"
        "Exclude_Recommended,ABE3_PartyID,ABE3_ResponseStyle,_Generation_Source\n"
        "R_1,1,PILOT_S0000000007,PILOT,7,4.6,0,3,acquiescent,llm\n"
    )
    out = sanitize_diagnostics_csv(csv_text.encode()).decode()
    header = out.splitlines()[0].split(",")
    for column in _FORBIDDEN_DIAGNOSTIC_COLUMNS:
        assert column not in header, f"{column} reached the user's diagnostics file"
    assert "PartyID" in header and "ABE3_PartyID" not in header
    assert ["ResponseId", "PARTICIPANT_ID", "Trust_mean", "Exclude_Recommended", "PartyID"] == header
    assert find_simulation_internals(out) == []


def test_a_broken_diagnostics_file_is_dropped_not_shipped():
    assert sanitize_diagnostics_csv(b"") == b""
    assert sanitize_diagnostics_csv(b"SIMULATION_SEED\n7\n") == b""


def test_public_run_id_does_not_carry_the_seed():
    assert public_run_id("PILOT_S0000000007").startswith("PILOT_")
    assert "0000000007" not in public_run_id("PILOT_S0000000007")
    assert public_run_id("FINAL_S0000012345") != public_run_id("FINAL_S0000012346")
    assert public_run_id("") == ""


# ──────────────────────────────────────────────────────────────────────────────
# Prose filters
# ──────────────────────────────────────────────────────────────────────────────

def test_codebook_drops_the_effect_size_answer_key():
    rule = "-" * 70
    text = "\n".join([
        "Study: Trust study", "Detected Domains: political_psychology", "Mode: PILOT", "",
        rule, "SURVEY COLUMNS (Simulated_Data.csv)", rule, "",
        "Trust_1", "    Trust item 1 (1-7)", "",
        rule, "EXPECTED EFFECT SIZES", rule, "", "  Trust:  > , Cohen's d = 0.5", "",
        rule, "EXCLUSION CRITERIA", rule, "", "  Min completion time: 60s", "",
    ])
    out = sanitize_codebook(text).decode()
    assert "Trust item 1 (1-7)" in out, "the codebook lost the column it exists to document"
    assert "EXPECTED EFFECT SIZES" not in out
    assert "Cohen's d = 0.5" not in out
    assert "EXCLUSION CRITERIA" not in out
    assert "Detected Domains" not in out
    assert find_simulation_internals(out) == []


def test_study_summary_drops_the_simulation_sections_and_keeps_the_analysis_guidance():
    markdown = "\n".join([
        "# Study Summary: Trust study", "",
        "## Generation Information", "", "| Field | Value |", "|---|---|",
        "| **Random Seed** | 7 |", "| **Generation Method** | Adaptive Behavioral Engine 3.0 |",
        "| **Run ID** | `PILOT_S0000000007` |", "",
        "## Simulation Intelligence Report", "",
        "### Condition Effects Strategy", "", "| Variable | Intended d |", "|---|---|", "| Trust | +0.50 |", "",
        "## Persona Distribution (Simulated Response Styles)", "",
        "| Persona | Share |", "|---|---|", "| Careless_Responder | 5% |", "",
        "## Analysis Recommendations", "",
        "**Primary analysis: One-way ANOVA with post-hoc pairwise tests**", "",
        "- Exclusion rates may indicate stricter criteria or more careless responders", "",
        "| Measure | Small | Medium | Large |", "|---|---|---|---|",
        "| Cohen's d | 0.20 | 0.50 | 0.80 |", "",
    ])
    out = sanitize_study_summary(markdown, run_id="PILOT_S0000000007", public_id="PILOT_ABCDEF0123")
    assert "Simulation Intelligence Report" not in out
    assert "Persona Distribution" not in out
    assert "Intended d" not in out
    assert "Random Seed" not in out
    assert "Adaptive Behavioral Engine" not in out
    assert "PILOT_S0000000007" not in out
    # The analysis guidance survives, minus the one offending sentence.
    assert "One-way ANOVA with post-hoc pairwise tests" in out
    assert "| Cohen's d | 0.20 | 0.50 | 0.80 |" in out
    assert "careless responders" not in out
    assert find_simulation_internals(out) == []


# ──────────────────────────────────────────────────────────────────────────────
# The package
# ──────────────────────────────────────────────────────────────────────────────

def test_owner_only_files_never_enter_the_user_package():
    files = {name: b"Persona roster, cohens_d 0.5, seed 7" for name in OWNER_ONLY_FILE_NAMES}
    files["Simulated_Data.csv"] = b"CONDITION,Trust_1\nControl,5\n"
    user = build_user_package(files, {"run_id": "PILOT_S0000000007", "study_title": "T"})
    for name in OWNER_ONLY_FILE_NAMES:
        assert name not in user, f"{name} reached the user's download"
    assert "Simulated_Data.csv" in user


def test_the_full_package_is_never_passed_through_unchanged():
    files = {
        "Simulated_Data.csv": b"CONDITION,Trust_1\nControl,5\n",
        "Metadata.json": json.dumps({"run_id": "PILOT_S0000000007", "seed": 7,
                                     "personas_used": ["careless_responder"]}).encode(),
    }
    user = build_user_package(files, {"run_id": "PILOT_S0000000007", "seed": 7,
                                      "personas_used": ["careless_responder"]})
    assert "Metadata.json" not in user, "the engine's own metadata file was delivered"
    assert "Study_Metadata.json" in user
    assert find_simulation_internals(user["Study_Metadata.json"].decode()) == []


def test_html_summary_is_built_from_the_sanitized_markdown():
    files = {"User_Study_Summary.md": "\n".join([
        "# S", "", "## Simulation Intelligence Report", "", "Personas were assigned.", "",
        "## Quick data checks", "", "N = 60", "",
    ]).encode()}
    captured = {}

    def _to_html(markdown: str, title: str) -> str:
        captured["markdown"] = markdown
        return f"<html><title>{title}</title><body>{markdown}</body></html>"

    user = build_user_package(files, {"study_title": "S"}, markdown_to_html=_to_html)
    assert "Simulation Intelligence Report" not in captured["markdown"]
    assert "User_Study_Summary.html" in user
    assert find_simulation_internals(user["User_Study_Summary.html"].decode()) == []


# ──────────────────────────────────────────────────────────────────────────────
# Wiring in app.py
# ──────────────────────────────────────────────────────────────────────────────

def test_app_hands_the_download_button_the_user_package_not_the_owner_zip():
    source = _APP_PY.read_text(encoding="utf-8")
    assert "_build_user_package(" in source, "app.py no longer derives a user package"
    # The owner ZIP must never be the download. `last_zip` feeds the download button and both
    # user-facing email buttons, so a plain assignment of `zip_bytes` is the regression.
    assert not re.search(r'st\.session_state\["last_zip"\]\s*=\s*zip_bytes\s*$', source, re.MULTILINE), (
        "app.py assigns the owner ZIP to last_zip; the download would carry simulation internals"
    )
    assert "from utils.user_package import" in source


# ──────────────────────────────────────────────────────────────────────────────
# End to end through the real engine
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_a_real_run_delivers_no_simulation_internals():
    """Build the owner package from a real run, derive the user package, scan every file."""
    import pandas as pd  # noqa: F401 - the engine needs it

    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    from utils.instructor_report import InstructorReportGenerator
    from utils.schema_validator import validate_schema
    from utils.survey_builder import ParsedDesign, SurveyDescriptionParser

    parser = SurveyDescriptionParser()
    conditions, _ = parser.parse_conditions("Control, Trump supporter match, Trump hater match")
    design = ParsedDesign(
        conditions=conditions,
        scales=parser.parse_scales("Trust scale, 5 items, 1-7 Likert\n"
                                   "Amount allocated in dictator game, 1 item, 0-100"),
        open_ended=parser.parse_open_ended("Why did you allocate that amount?"),
        design_type="between", sample_size=40,
        study_title="Political Identity Dictator Game",
        study_description="Partisan identity and economic allocation in a dictator game.",
    )
    inferred = parser.build_inferred_design(design)
    engine = EnhancedSimulationEngine(
        study_title=design.study_title, study_description=design.study_description,
        sample_size=40, conditions=inferred["conditions"], factors=inferred.get("factors", []),
        scales=inferred["scales"], additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, seed=7,
        effect_sizes=[{"condition": "Trump supporter match", "variable": "Trust", "cohens_d": 0.5}],
    )
    if getattr(engine, "llm_generator", None) is not None:
        engine.llm_generator.disable_permanently("secrecy test - no network")
    df, metadata = engine.generate()

    schema_results = validate_schema(df=df, expected_conditions=inferred["conditions"],
                                     expected_scales=inferred["scales"], expected_n=40)
    summary = InstructorReportGenerator().generate_markdown_report(
        df=df, metadata=metadata, schema_validation=schema_results, prereg_text="",
        team_info={"team_name": "", "team_members": ""})

    files = {
        "Simulated_Data.csv": df.to_csv(index=False).encode(),
        "Metadata.json": json.dumps(metadata, indent=2, default=str).encode(),
        "Data_Codebook_Handbook.txt": engine.generate_explainer().encode(),
        "Schema_Validation.json": json.dumps(schema_results, indent=2, default=str).encode(),
        "User_Study_Summary.md": summary.encode(),
        "R_Prepare_Data.R": engine.generate_r_export(df).encode(),
        "Python_Prepare_Data.py": engine.generate_python_export(df).encode(),
        "Julia_Prepare_Data.jl": engine.generate_julia_export(df).encode(),
        "SPSS_Prepare_Data.sps": engine.generate_spss_export(df).encode(),
        "Stata_Prepare_Data.do": engine.generate_stata_export(df).encode(),
        "INSTRUCTOR_Detailed_Analysis.md": b"owner only: cohens_d, personas, seed",
    }
    try:
        from utils.qualtrics_export import (
            build_qualtrics_export, export_to_csv_bytes, to_qualtrics_raw_csv,
        )
        qx_df, qx_diag = build_qualtrics_export(df, metadata)
        files["Simulated_Data.csv"] = export_to_csv_bytes(qx_df)
        files["Simulation_Diagnostics.csv"] = qx_diag.to_csv(index=False).encode()
        files["Simulated_Data_Qualtrics_Raw.csv"] = to_qualtrics_raw_csv(
            qx_df, metadata.get("column_descriptions", {}))
    except Exception as err:  # pragma: no cover - the export is covered by its own tests
        pytest.skip(f"Qualtrics export unavailable: {err}")

    # Sanity: the owner package really does carry internals, so a passing user scan means something.
    owner_blob = b"\n".join(files.values()).decode("utf-8", "replace")
    assert find_simulation_internals(owner_blob), "the owner package no longer carries internals"

    user = build_user_package(files, metadata)
    assert "Simulated_Data.csv" in user and "Study_Metadata.json" in user
    leaks = {name: find_simulation_internals(data.decode("utf-8", "replace"))
             for name, data in user.items()}
    leaks = {name: hits for name, hits in leaks.items() if hits}
    assert not leaks, f"simulation internals in the user download: {leaks}"

    assert "INSTRUCTOR_Detailed_Analysis.md" not in user
    assert str(metadata.get("run_id", "")) not in user["Study_Metadata.json"].decode()
    # The data itself is delivered intact.
    assert user["Simulated_Data.csv"] == files["Simulated_Data.csv"]


def test_internal_columns_are_stripped_from_the_delivered_csv():
    from utils.user_package import strip_internal_columns

    clean = b'StartDate,CONDITION,Trust_1\n2026-01-01,Control,5\n'
    assert strip_internal_columns(clean) == clean, "a clean CSV must pass through byte-identical"

    leaky = (b"CONDITION,Trust_1,SIMULATION_SEED,ABE3_ResponseStyle,_Generation_Source\n"
             b"Control,5,7,acquiescent,llm\n")
    out = strip_internal_columns(leaky).decode()
    assert out.splitlines()[0] == "CONDITION,Trust_1"
    assert find_simulation_internals(out) == []

    # A user's own scale that merely reads like an internal must survive.
    user_named = b"CONDITION,Traits_1,Response_Style_1\nControl,5,3\n"
    assert strip_internal_columns(user_named) == user_named


def test_a_three_row_qualtrics_header_keeps_its_shape():
    from utils.user_package import strip_internal_columns

    raw = ("CONDITION,Trust_1,SIMULATION_SEED\n"
           "Condition,Trust item 1,Seed\n"
           '{"ImportId":"c"},{"ImportId":"t"},{"ImportId":"s"}\n'
           "Control,5,7\n").encode()
    rows = strip_internal_columns(raw).decode().strip().split("\n")
    assert len(rows) == 4
    assert all(len(next(__import__("csv").reader([r]))) == 2 for r in rows)
