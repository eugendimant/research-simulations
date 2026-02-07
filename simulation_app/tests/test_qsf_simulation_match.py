#!/usr/bin/env python3
"""
End-to-end QSF → Simulation verification test harness.

For every QSF file:
1. Parse the QSF
2. Extract detected scales, conditions, open-ended questions
3. Run a trial simulation
4. Verify: all QSF-detected scale columns exist in the DataFrame
5. Verify: no mystery scale columns exist that aren't from the QSF
6. Verify: all values are within the specified scale range
7. Report any mismatches

Usage:
    python3 simulation_app/tests/test_qsf_simulation_match.py
"""
import sys
import os
import json
import traceback
import io
import zipfile

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine


# ---------------------------------------------------------------------------
# Helper: replicate app.py's _preview_to_engine_inputs + _normalize_scale_specs
# ---------------------------------------------------------------------------

def _safe_int(val, default):
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def preview_to_engine_inputs(preview):
    """Replicate the app.py conversion from QSFPreviewResult to engine inputs."""
    conditions = [
        str(c).replace('\xa0', ' ').strip()
        for c in (preview.detected_conditions or [])
        if str(c).replace('\xa0', ' ').strip()
    ]
    if not conditions:
        conditions = ["Condition_A"]

    scales = []
    seen_var = set()       # Deduplicate by variable_name
    seen_display = set()   # Also deduplicate by display_name to prevent column collisions
    for s in (preview.detected_scales or []):
        name = str(s.get("variable_name", s.get("name", "Scale"))).strip() or "Scale"
        display_name = str(s.get("name", name)).strip() or name
        name_key = name.lower().replace(" ", "_").replace("-", "_")
        display_key = display_name.lower().replace(" ", "_").replace("-", "_")
        if name_key in seen_var or display_key in seen_display:
            continue
        seen_var.add(name_key)
        seen_display.add(display_key)

        # Check both "num_items" and "items" keys
        raw_items = s.get("num_items")
        if raw_items is None:
            raw_items = s.get("items")
        num_items = _safe_int(raw_items, 5)

        scale_points = _safe_int(s.get("scale_points"), 7)

        scales.append({
            "name": display_name,
            "variable_name": name.replace(" ", "_"),
            "num_items": max(1, num_items),
            "scale_points": max(2, min(1001, scale_points)),
            "reverse_items": s.get("reverse_items", []) or [],
            "_validated": True,
        })

    if not scales:
        scales = [{"name": "Main_DV", "variable_name": "Main_DV", "num_items": 5,
                    "scale_points": 7, "reverse_items": [], "_validated": True}]

    open_ended_details = getattr(preview, "open_ended_details", None) or []
    study_context = getattr(preview, "study_context", None) or {}

    return {
        "conditions": conditions,
        "scales": scales,
        "open_ended_details": open_ended_details,
        "study_context": study_context,
        "condition_visibility_map": getattr(preview, "condition_visibility_map", {}) or {},
    }


def extract_qsf_payload(uploaded_bytes):
    """Return JSON bytes from a QSF file (supports raw JSON or ZIP wrappers)."""
    if zipfile.is_zipfile(io.BytesIO(uploaded_bytes)):
        with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
            candidates = [n for n in zf.namelist() if n.lower().endswith((".qsf", ".json"))]
            if not candidates:
                raise ValueError("ZIP did not contain a .qsf or .json file.")
            return zf.read(candidates[0])
    return uploaded_bytes


def run_trial_simulation(inputs, sample_size=30):
    """Run a trial simulation and return the DataFrame + metadata."""
    engine = EnhancedSimulationEngine(
        study_title="QSF Test",
        study_description="Testing QSF-to-simulation match",
        sample_size=sample_size,
        conditions=inputs["conditions"],
        factors=[],
        scales=inputs["scales"],
        additional_vars=[],
        demographics={"age_mean": 35, "age_sd": 10},
        attention_rate=0.95,
        random_responder_rate=0.02,
        effect_sizes=[],
        open_ended_questions=inputs.get("open_ended_details", []),
        study_context=inputs.get("study_context", {}),
        seed=42,
        precomputed_visibility=inputs.get("condition_visibility_map", {}),
    )
    return engine.generate()


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------

def verify_simulation(qsf_file_path, verbose=False):
    """
    Parse a QSF file, run a simulation, and verify that:
    1. All expected scale columns exist
    2. All scale values are within bounds
    3. No unexpected scale-like columns exist
    """
    issues = []
    file_name = os.path.basename(qsf_file_path)

    # Step 1: Parse QSF
    try:
        with open(qsf_file_path, "rb") as f:
            raw = f.read()
        payload = extract_qsf_payload(raw)
    except Exception as e:
        return [{"file": file_name, "issue": f"PARSE_LOAD_ERROR: {e}"}]

    try:
        parser = QSFPreviewParser()
        preview = parser.parse(payload)
    except Exception as e:
        return [{"file": file_name, "issue": f"PARSE_ERROR: {e}"}]

    if not preview.success and not preview.detected_scales:
        return [{"file": file_name, "issue": "PARSE_FAILED: No scales detected", "severity": "info"}]

    # Step 2: Convert to engine inputs
    try:
        inputs = preview_to_engine_inputs(preview)
    except Exception as e:
        return [{"file": file_name, "issue": f"CONVERT_ERROR: {e}"}]

    # Step 3: Run trial simulation
    try:
        df, metadata = run_trial_simulation(inputs)
    except Exception as e:
        tb = traceback.format_exc()
        # Get last 3 lines of traceback for location info
        tb_lines = [l.strip() for l in tb.strip().split('\n') if l.strip()]
        loc = tb_lines[-3] if len(tb_lines) >= 3 else ""
        return [{"file": file_name, "issue": f"SIMULATION_ERROR: {type(e).__name__}: {e}\n    at: {loc}"}]

    # Step 4: Verify - all expected scale columns exist
    # The engine uses scale["name"] (not variable_name) for column generation
    expected_scale_cols = set()
    scale_bounds = {}  # col_name -> (min, max)
    for scale in inputs["scales"]:
        # Engine uses "name" field for column prefix, matching the generate() method:
        #   scale_name_raw = str(scale.get("name", "")).strip()
        #   scale_name = scale_name_raw.replace(" ", "_")
        #   col_name = f"{scale_name}_{item_num}"
        scale_name = scale["name"].strip().replace(" ", "_")
        num_items = scale["num_items"]
        scale_points = scale["scale_points"]
        for item in range(1, num_items + 1):
            col = f"{scale_name}_{item}"
            expected_scale_cols.add(col)
            scale_bounds[col] = (1, scale_points)

    # Check for missing columns
    for col in expected_scale_cols:
        if col not in df.columns:
            issues.append({
                "file": file_name,
                "issue": f"MISSING_COLUMN: '{col}' expected from scale spec but not in DataFrame",
                "severity": "critical",
            })

    # Step 5: Verify - all values within bounds
    for col, (exp_min, exp_max) in scale_bounds.items():
        if col not in df.columns:
            continue
        actual_min = int(df[col].min())
        actual_max = int(df[col].max())
        if actual_min < exp_min:
            issues.append({
                "file": file_name,
                "issue": f"BOUNDS_VIOLATION: '{col}' min={actual_min} < expected {exp_min}",
                "severity": "critical",
            })
        if actual_max > exp_max:
            issues.append({
                "file": file_name,
                "issue": f"BOUNDS_VIOLATION: '{col}' max={actual_max} > expected {exp_max}",
                "severity": "critical",
            })

    # Step 6: Verify - no unexpected DV-like columns
    # Known structural columns that are NOT from scales
    structural_cols = {
        "PARTICIPANT_ID", "CONDITION", "RUN_ID", "SIMULATION_RUN_ID",
        "SIMULATION_MODE", "SIMULATION_SEED",
        "Age", "Gender",
        "Completion_Time_Seconds", "Attention_Pass_Rate", "Max_Straight_Line",
        "Flag_Speed", "Flag_Attention", "Flag_StraightLine",
        "Exclude_Recommended", "Attention_Check_1", "AI_Mentioned_Check", "Hedonic_Utilitarian",
        "Attention_Total_Correct", "Max_StraightLine",
        "_PERSONA",
    }

    # Build set of all expected OE column names (including renamed ones)
    oe_col_names = set()
    for q in inputs.get("open_ended_details", []):
        col = str(q.get("name", "Open_Response")).replace(" ", "_")
        oe_col_names.add(col)
        oe_col_names.add(f"OE_{col}")  # Possible renamed version

    for col in df.columns:
        if col in structural_cols:
            continue
        if col in expected_scale_cols:
            continue
        if col in oe_col_names:
            continue
        # Open-ended columns are text dtype
        if df[col].dtype == object:
            continue
        # Renamed OE columns from collision avoidance
        if col.startswith("OE_") or col.endswith("_OE"):
            continue
        # This is a numeric column not in our expected set - potential ghost column
        # But only flag if it looks like a scale item (has _N suffix with a number)
        import re
        if re.match(r'.+_\d+$', col):
            issues.append({
                "file": file_name,
                "issue": f"UNEXPECTED_COLUMN: '{col}' is numeric and looks like a scale item but not in QSF spec",
                "severity": "warning",
            })

    # Step 7: Check scale verification report
    verification = metadata.get("scale_verification", [])
    for v in verification:
        if v.get("status") not in ("OK", None):
            issues.append({
                "file": file_name,
                "issue": f"VERIFICATION_REPORT: Scale '{v['name']}' status={v['status']} "
                         f"(utilization={v.get('range_utilization_pct', '?')}%)",
                "severity": "warning",
            })

    if verbose and not issues:
        print(f"  OK: {file_name} - {len(inputs['scales'])} scales, "
              f"{len(expected_scale_cols)} columns, {len(df)} rows")

    return issues


def run_full_verification(qsf_dir, verbose=True, max_files=None):
    """Run verification against all QSF files in a directory."""
    import glob

    qsf_files = sorted(glob.glob(os.path.join(qsf_dir, "*.qsf")))
    if max_files:
        qsf_files = qsf_files[:max_files]

    total = len(qsf_files)
    passed = 0
    failed = 0
    errored = 0
    all_issues = []

    critical_issues = []
    warning_issues = []
    info_issues = []

    print(f"Testing {total} QSF files...\n")

    for idx, qsf_file in enumerate(qsf_files):
        file_name = os.path.basename(qsf_file)
        try:
            issues = verify_simulation(qsf_file, verbose=verbose)
            if not issues:
                passed += 1
            else:
                severity_set = {i.get("severity", "critical") for i in issues}
                if "critical" in severity_set:
                    failed += 1
                else:
                    passed += 1  # Warnings/info only = pass
                for issue in issues:
                    sev = issue.get("severity", "critical")
                    if sev == "critical":
                        critical_issues.append(issue)
                    elif sev == "warning":
                        warning_issues.append(issue)
                    else:
                        info_issues.append(issue)
                all_issues.extend(issues)
        except Exception as e:
            errored += 1
            all_issues.append({
                "file": file_name,
                "issue": f"UNHANDLED_ERROR: {type(e).__name__}: {e}",
                "severity": "critical",
            })

        # Progress indicator
        if (idx + 1) % 20 == 0 or idx + 1 == total:
            print(f"  Progress: {idx+1}/{total} ({passed} passed, {failed} failed, {errored} errored)")

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed, {errored} errored out of {total}")
    print(f"{'='*70}")

    if critical_issues:
        print(f"\n--- CRITICAL ISSUES ({len(critical_issues)}) ---")
        for issue in critical_issues[:50]:
            print(f"  [{issue['file']}] {issue['issue']}")

    if warning_issues:
        print(f"\n--- WARNINGS ({len(warning_issues)}) ---")
        for issue in warning_issues[:30]:
            print(f"  [{issue['file']}] {issue['issue']}")

    if info_issues:
        print(f"\n--- INFO ({len(info_issues)}) ---")
        for issue in info_issues[:20]:
            print(f"  [{issue['file']}] {issue['issue']}")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "critical_issues": critical_issues,
        "warning_issues": warning_issues,
        "info_issues": info_issues,
    }


if __name__ == "__main__":
    qsf_dir = os.path.join(os.path.dirname(__file__), "..", "example_files")
    results = run_full_verification(qsf_dir)
    sys.exit(0 if results["failed"] == 0 and results["errored"] == 0 else 1)
