from __future__ import annotations
import sys
import numpy as np

def main() -> int:
    sys.path.insert(0, ".")
    from socsim.games.registry import GAME_REGISTRY
    from socsim.persona import Persona
    rng = np.random.default_rng(0)
    a = Persona(id="A", params={"strategic_depth": 2.0, "competitiveness": 0.5, "dishonesty": 0.3, "risk_aversion": 0.2, "trust": 0.3, "prosociality": 0.1, "norm_sensitivity": 0.2}, latent_class="base")
    b = Persona(id="B", params={"risk_aversion": 0.1, "trust": 0.1}, latent_class="base")
    for name in ["beauty_contest","common_pool_resource","tullock_contest","bribery_game"]:
        g = GAME_REGISTRY[name]()
        out = g.simulate_one(rng, a, None, {})
        assert isinstance(out.actions, dict)
    gh = GAME_REGISTRY["stag_hunt"]()
    out = gh.simulate_one(rng, a, b, {})
    assert out.actions["choice_A"] in ("C","D")
    assert out.actions["choice_B"] in ("C","D")
    print("verify_games_extra_ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
