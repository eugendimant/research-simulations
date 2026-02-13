import pandas as pd
from socsim.bench.moments import Moment, compute_targets

def test_grouped_moments_and_gini_share_corr_slope():
    df = pd.DataFrame({
        "x": [1,2,3,4,5,6],
        "y": [1,1,2,2,3,3],
        "g": ["A","A","A","B","B","B"],
        "treat": [0,0,0,1,1,1]
    })
    moms = [
        Moment(id="g_mean", kind="by_mean", column="x", by="g"),
        Moment(id="g_q50", kind="by_quantile", column="x", by="g", q=0.5),
        Moment(id="g_share", kind="by_share", column="x", by="g", op="gt", value=3),
        Moment(id="gini", kind="gini", column="x"),
        Moment(id="corr", kind="corr", x="x", y="y"),
        Moment(id="slope", kind="slope", by="x", column="y"),
        Moment(id="diff", kind="diff_in_means", column="x", treat="treat"),
    ]
    out = compute_targets(df, moms)
    assert set(out["g_mean"].keys()) == {"A","B"}
    assert out["g_q50"]["A"] == 2.0
    assert out["g_share"]["A"] == 0.0
    assert out["g_share"]["B"] == 1.0
    assert 0.0 <= out["gini"] <= 1.0
    assert out["diff"] > 0
