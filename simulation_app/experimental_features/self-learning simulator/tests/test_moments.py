import pandas as pd
from socsim.bench.moments import Moment, compute_targets

def test_moments_basic():
    df = pd.DataFrame({"offer":[3,4,5,6], "treatment":["A","A","B","B"], "round":[1,2,1,2]})
    moms = [
        Moment(id="mean_offer", kind="mean", column="offer"),
        Moment(id="q50_offer", kind="quantile", column="offer", q=0.5),
        Moment(id="diff_offer", kind="diff_in_means", column="offer", treat="treatment"),
        Moment(id="traj_offer", kind="trajectory_mean", column="offer", by="round"),
    ]
    t = compute_targets(df, moms)
    assert abs(t["mean_offer"] - 4.5) < 1e-9
    assert abs(t["q50_offer"] - 4.5) < 1e-9
    assert abs(t["diff_offer"] - ((5+6)/2 - (3+4)/2)) < 1e-9
    assert t["traj_offer"]["1"] == 4.0
    assert t["traj_offer"]["2"] == 5.0
