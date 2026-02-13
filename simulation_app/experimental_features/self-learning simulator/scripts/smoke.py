from __future__ import annotations
import sys
import numpy as np

def main() -> int:
    sys.path.insert(0, ".")
    import socsim
    from socsim.simulator import Simulator
    from socsim.games.registry import make_game
    from socsim.surveys.runner import SurveySpec, simulate_survey
    from socsim.persona import Persona

    rng = np.random.default_rng(0)
    sim = Simulator(seed=0)
    # basic game simulation
    g = make_game("dictator")
    a = Persona(id="A", params={"prosociality": 0.2}, latent_class="base")
    out = g.simulate_one(rng, a, None, {"endowment": 10})
    assert "give" in out.actions

    # new games compile
    for nm in ["beauty_contest","common_pool_resource","tullock_contest","bribery_game","stag_hunt"]:
        make_game(nm)

    # survey primitives
    r = simulate_survey(rng=np.random.default_rng(1), persona={"attitude": 0.1}, context={"treated": True, "endorsement": True}, spec=SurveySpec("likert", {"k": 7}))
    assert 1 <= r["resp::likert"] <= 7

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
