from __future__ import annotations
import sys, random

def main() -> int:
    sys.path.insert(0, ".")
    from socsim.surveys.runner import SurveySpec, simulate_survey
    rng = random.Random(0)
    persona = {"attitude": 0.3}
    context = {"treated": True, "endorsement": True}
    out1 = simulate_survey(rng, persona, context, SurveySpec("likert", {"k": 7}))
    assert 1 <= out1["resp::likert"] <= 7
    out2 = simulate_survey(rng, persona, context, SurveySpec("list_experiment", {}))
    assert isinstance(out2["resp::count"], int)
    out3 = simulate_survey(rng, persona, context, SurveySpec("randomized_response", {"truth": True, "p_truth": 0.7}))
    assert out3["resp::rr"] in (0,1)
    out4 = simulate_survey(rng, persona, context, SurveySpec("endorsement", {"base_support": 0.5, "endorsement_shift": 0.1}))
    assert out4["resp::support"] in (0,1)
    print("verify_surveys_ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
