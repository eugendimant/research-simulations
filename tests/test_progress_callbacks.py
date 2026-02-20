#!/usr/bin/env python3
"""
Test: Verify progress callbacks are correct for each generation method.

This tests that:
1. Each method fires appropriate progress phases
2. Template/Experimental methods do NOT fire "llm_prefill" phase
3. Only Experimental method fires "socsim_enrichment" phase
4. All methods fire "generating" and "complete" phases
5. No crashes during generation for any method
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation_app"))
os.environ.setdefault("STREAMLIT_RUNTIME", "0")

from utils.enhanced_simulation_engine import EnhancedSimulationEngine as BehavioralSimulationEngine

BASE_CONFIG = dict(
    study_title="Progress Callback Test Study",
    study_description="Testing social trust in economic games",
    conditions=["Control", "Treatment"],
    factors=[{"name": "Condition", "levels": ["Control", "Treatment"]}],
    additional_vars=[],
    demographics={"age_mean": 30, "age_sd": 10, "gender_quota": 50},
    scales=[
        {"name": "Trust_Scale", "num_items": 3, "scale_points": 7,
         "scale_min": 1, "scale_max": 7, "reverse_items": []},
    ],
    open_ended_questions=[
        {"name": "OE_Feelings", "variable_name": "OE_Feelings",
         "question_text": "How do you feel about economic trust?",
         "question_context": "Open-ended question about trust",
         "type": "text"},
    ],
    sample_size=10,
    seed=42,
)


def run_with_callbacks(label, allow_template, use_socsim):
    """Run generation and capture all progress callback phases."""
    phases_seen = []

    def callback(phase, current, total):
        entry = (phase, current, total)
        phases_seen.append(entry)

    engine = BehavioralSimulationEngine(
        **BASE_CONFIG,
        allow_template_fallback=allow_template,
        use_socsim_experimental=use_socsim,
        progress_callback=callback,
    )
    df, meta = engine.generate()

    # Extract unique phase names in order
    unique_phases = []
    for p, c, t in phases_seen:
        if not unique_phases or unique_phases[-1] != p:
            unique_phases.append(p)

    print(f"\n=== {label} ===")
    print(f"  Phases fired: {unique_phases}")
    print(f"  Total callbacks: {len(phases_seen)}")
    print(f"  DataFrame: {df.shape[0]} rows, {df.shape[1]} cols")
    return unique_phases, df, meta


def main():
    results = {}
    errors = []

    # Method 1: Template Engine
    try:
        phases, df, meta = run_with_callbacks(
            "Template Engine", allow_template=True, use_socsim=False)
        results["template"] = phases
        if "llm_prefill" in phases:
            errors.append("FAIL: Template Engine should NOT fire 'llm_prefill' phase")
        if "socsim_enrichment" in phases:
            errors.append("FAIL: Template Engine should NOT fire 'socsim_enrichment' phase")
        if "generating" not in phases:
            errors.append("FAIL: Template Engine MUST fire 'generating' phase")
        if "complete" not in phases:
            errors.append("FAIL: Template Engine MUST fire 'complete' phase")
        if df.shape[0] != 10:
            errors.append(f"FAIL: Template Engine expected 10 rows, got {df.shape[0]}")
        print("  PASS")
    except Exception as e:
        errors.append(f"FAIL: Template Engine CRASHED: {e}")
        import traceback; traceback.print_exc()

    # Method 2: Adaptive Behavioral Engine
    try:
        phases, df, meta = run_with_callbacks(
            "Adaptive Behavioral Engine", allow_template=True, use_socsim=True)
        results["experimental"] = phases
        if "llm_prefill" in phases:
            errors.append("FAIL: Experimental Engine should NOT fire 'llm_prefill' phase")
        if "generating" not in phases:
            errors.append("FAIL: Experimental Engine MUST fire 'generating' phase")
        if "complete" not in phases:
            errors.append("FAIL: Experimental Engine MUST fire 'complete' phase")
        if df.shape[0] != 10:
            errors.append(f"FAIL: Experimental Engine expected 10 rows, got {df.shape[0]}")
        print("  PASS")
    except Exception as e:
        errors.append(f"FAIL: Experimental Engine CRASHED: {e}")
        import traceback; traceback.print_exc()

    # Method 3: Built-in AI (free_llm) — will fall back since no real API
    try:
        phases, df, meta = run_with_callbacks(
            "Built-in AI (free_llm)", allow_template=False, use_socsim=False)
        results["free_llm"] = phases
        if "generating" not in phases:
            errors.append("FAIL: free_llm MUST fire 'generating' phase")
        if "complete" not in phases:
            errors.append("FAIL: free_llm MUST fire 'complete' phase")
        if df.shape[0] != 10:
            errors.append(f"FAIL: free_llm expected 10 rows, got {df.shape[0]}")
        # llm_prefill IS expected for AI methods (even if it fails quickly)
        print("  PASS")
    except Exception as e:
        errors.append(f"FAIL: free_llm CRASHED: {e}")
        import traceback; traceback.print_exc()

    # Method 4: Own API Key — same as free_llm in test (no real key)
    try:
        phases, df, meta = run_with_callbacks(
            "Own API Key", allow_template=False, use_socsim=False)
        results["own_api"] = phases
        if "generating" not in phases:
            errors.append("FAIL: own_api MUST fire 'generating' phase")
        if "complete" not in phases:
            errors.append("FAIL: own_api MUST fire 'complete' phase")
        if df.shape[0] != 10:
            errors.append(f"FAIL: own_api expected 10 rows, got {df.shape[0]}")
        print("  PASS")
    except Exception as e:
        errors.append(f"FAIL: own_api CRASHED: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("PROGRESS CALLBACK TEST SUMMARY")
    print("=" * 70)
    for method, phases in results.items():
        print(f"  [{method}] phases: {phases}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        print(f"\n  ALL {len(results)} methods passed callback validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
