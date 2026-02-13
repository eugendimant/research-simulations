from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

@dataclass
class Moment:
    id: str
    kind: str
    column: Optional[str] = None
    q: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None
    by: Optional[str] = None
    treat: Optional[str] = None
    op: Optional[str] = None
    value: Optional[float] = None
    x: Optional[str] = None
    y: Optional[str] = None

def _apply_filter(df: pd.DataFrame, flt: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not flt:
        return df
    out = df
    for k, v in flt.items():
        out = out[out[k] == v]
    return out

def _stable_group_key(key: Any) -> str:
    try:
        if isinstance(key, float) and float(key).is_integer():
            key = int(key)
    except Exception:
        pass
    return str(key)

def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)

def _share(series: pd.Series, op: str, value: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    if op == "gt":
        return float((s > value).mean())
    if op == "ge":
        return float((s >= value).mean())
    if op == "lt":
        return float((s < value).mean())
    if op == "le":
        return float((s <= value).mean())
    if op == "eq":
        return float((s == value).mean())
    raise ValueError(f"Unknown share op: {op}")

def compute_moment(df: pd.DataFrame, m: Moment) -> Any:
    d = _apply_filter(df, m.filter)
    kind = m.kind

    if kind == "mean":
        return float(d[m.column].mean())
    if kind == "std":
        return float(d[m.column].std(ddof=1))
    if kind == "var":
        return float(d[m.column].var(ddof=1))
    if kind == "quantile":
        return float(d[m.column].quantile(float(m.q)))
    if kind == "gini":
        return _gini(d[m.column].to_numpy())
    if kind == "share":
        if m.op is None or m.value is None:
            raise ValueError("share requires op and value")
        return _share(d[m.column], m.op, float(m.value))
    if kind == "corr":
        x = m.x or m.column
        y = m.y
        if not x or not y:
            raise ValueError("corr requires x and y")
        dd = d[[x, y]].dropna()
        if len(dd) < 2:
            return float("nan")
        return float(dd[x].corr(dd[y]))

    if kind in ("by_mean", "by_std", "by_var", "by_gini", "by_share", "by_quantile"):
        if not m.by:
            raise ValueError(f"{kind} requires by")
        gb = d.groupby(m.by)
        out: Dict[str, float] = {}
        for key, sub in gb:
            k = _stable_group_key(key)
            if kind == "by_mean":
                out[k] = float(sub[m.column].mean())
            elif kind == "by_std":
                out[k] = float(sub[m.column].std(ddof=1))
            elif kind == "by_var":
                out[k] = float(sub[m.column].var(ddof=1))
            elif kind == "by_gini":
                out[k] = _gini(sub[m.column].to_numpy())
            elif kind == "by_share":
                if m.op is None or m.value is None:
                    raise ValueError("by_share requires op and value")
                out[k] = _share(sub[m.column], m.op, float(m.value))
            elif kind == "by_quantile":
                if m.q is None:
                    raise ValueError("by_quantile requires q")
                out[k] = float(sub[m.column].quantile(float(m.q)))
        return out

    if kind == "trajectory_mean":
        if not m.by:
            raise ValueError("trajectory_mean requires by")
        g = d.groupby(m.by)[m.column].mean().reset_index()
        out: Dict[str, float] = {}
        for _, row in g.iterrows():
            out[_stable_group_key(row[m.by])] = float(row[m.column])
        return out

    if kind == "diff_in_means":
        vals = sorted(list(d[m.treat].unique()))
        if len(vals) != 2:
            raise ValueError(f"diff_in_means needs exactly 2 treatment values; got {vals}")
        a = float(d[d[m.treat] == vals[0]][m.column].mean())
        b = float(d[d[m.treat] == vals[1]][m.column].mean())
        return float(b - a)

    if kind == "slope":
        if not m.by or not m.column:
            raise ValueError("slope requires by and column")
        dd = d[[m.by, m.column]].dropna()
        if len(dd) < 2:
            return float("nan")
        x = dd[m.by].astype(float).to_numpy()
        y = dd[m.column].astype(float).to_numpy()
        x = x - x.mean()
        denom = float((x**2).sum())
        if denom == 0:
            return 0.0
        beta = float((x * (y - y.mean())).sum() / denom)
        return beta

    raise ValueError(f"Unknown moment kind: {kind}")

def compute_targets(df: pd.DataFrame, moments: List[Moment]) -> Dict[str, Any]:
    return {m.id: compute_moment(df, m) for m in moments}

def moments_from_dict(payload: List[Dict[str, Any]]) -> List[Moment]:
    return [Moment(**p) for p in payload]
