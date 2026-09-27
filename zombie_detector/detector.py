"""
detector.py - zombie credit detector.

A "zombie" here is a project whose old credits sat untouched for years and then
moved in bulk (issued or retired), typically when a new buyer or price appeared.
In 2021-22 that buyer was the Toucan bridge.

Design principles:
  1. Point in time: a score for date T uses ONLY transactions dated <= T.
     Otherwise the backtest "sees the future" and its results are fiction.
  2. Explainable: no ML. Every score comes with reasons that can be shown to a
     client or an auditor.
  3. Relative: signals are compared with projects active at the same time
     (robust z-score: median/MAD), not fixed thresholds, because markets shift.
  4. The bridge marker (is_bridge) is NEVER a scoring input. It is only used as a
     label for validation; otherwise the detector would just learn the word "Toucan".

Usage:
    python detector.py --as-of 2022-03-01 --top 25
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

# Bridge / tokenization markers in retirement records (validation label, not a signal)
BRIDGE_RE = re.compile(
    r"0x[a-fA-F0-9]{40}|toucan|klimadao|c3 bridge|\bC3T-|moss\.earth|flowcarbon|tokeni[sz]",
    re.IGNORECASE,
)

# v0.3: the score uses ONE signal. Ablation on real data (2021-11..2022-06):
#   log_old_volume AUC 0.854, volume baseline 0.704, v0.2 (sum of 5 signals) 0.620.
# The other signals are kept as EXPLANATION tags, not ranking inputs:
# log_dormancy ranks in the wrong direction (AUC 0.42) but is precise at the top (P@20 0.40).
WEIGHTS = {"log_old_volume": 1.0}
FLAGS = ["vintage_age", "old_share", "log_dormancy", "flush_ratio", "log_burst"]

REASONS = {
    "vintage_age": "old vintages",
    "old_share": "high share of old credits",
    "log_dormancy": "long dormancy",
    "flush_ratio": "cleared its backlog",
    "log_burst": "sudden burst",
    "log_old_volume": "large volume of old credits moved",
    "log_volume": "high volume",
}


def load(data_dir: Path = DATA):
    credits = pd.read_parquet(data_dir / "credits.parquet")
    projects = pd.read_parquet(data_dir / "projects.parquet")
    text = (
        credits[["retirement_beneficiary", "retirement_note", "retirement_account"]]
        .fillna("").astype(str).agg(" ".join, axis=1)
    )
    credits["is_bridge"] = (credits["transaction_type"] == "retirement") & text.str.contains(BRIDGE_RE)
    return credits, projects


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not mad or np.isnan(mad):
        std = s.std()
        mad = std if std and not np.isnan(std) else 1.0
    return ((s - med) / mad).clip(-3, 6)


def _episodes(c: pd.DataFrame, ids, run_gap_days: int) -> pd.DataFrame:
    """Start of the current activity episode, and the quiet period before it, per project.

    Episode = a run of transactions with gaps < run_gap_days.
    v0.1 bug: dormancy was measured from the last transaction before the window, so
    a bridging run lasting several months "woke itself up" and the signal inverted.
    """
    d = (c.loc[c["project_id"].isin(ids), ["project_id", "transaction_date"]]
         .drop_duplicates().sort_values(["project_id", "transaction_date"]))
    d["gap"] = d.groupby("project_id")["transaction_date"].diff().dt.days
    starts = d[d["gap"].isna() | (d["gap"] > run_gap_days)]
    last = starts.groupby("project_id").last()
    return last.rename(columns={"transaction_date": "episode_start", "gap": "quiet_days"})


def features(credits, projects, as_of, window_days=90, history_years=3,
             old_age=6, min_volume=1_000, run_gap_days=60):
    """Signals for every project with activity in the window (as_of - window, as_of]."""
    as_of = pd.Timestamp(as_of, tz="UTC")
    start = as_of - pd.Timedelta(days=window_days)
    hist_start = start - pd.Timedelta(days=365 * history_years)

    c = credits[credits["transaction_date"] <= as_of]
    past, win = c[c["transaction_date"] < start], c[c["transaction_date"] >= start]
    if win.empty:
        return pd.DataFrame()

    g = win.groupby("project_id")
    f = pd.DataFrame({
        "win_issued": win[win.transaction_type == "issuance"].groupby("project_id")["quantity"].sum(),
        "win_retired": win[win.transaction_type == "retirement"].groupby("project_id")["quantity"].sum(),
        "first_win_txn": g["transaction_date"].min(),
        "label_bridge": g["is_bridge"].any() if "is_bridge" in win else False,
        "win_bridged": win[win["is_bridge"]].groupby("project_id")["quantity"].sum()
        if "is_bridge" in win else 0,
    }).fillna({"win_issued": 0, "win_retired": 0, "win_bridged": 0})
    f["bridge_share"] = (f["win_bridged"] / f["win_retired"].replace(0, np.nan)).fillna(0)
    f["win_volume"] = f["win_issued"] + f["win_retired"]
    f = f[f["win_volume"] >= min_volume]
    if f.empty:
        return f

    # 1) vintage age (quantity-weighted) and share of old credits
    w = win[win["project_id"].isin(f.index) & win["vintage"].notna()].copy()
    w["age"] = w["transaction_date"].dt.year - w["vintage"].astype(int)
    w["qa"] = w["quantity"] * w["age"]
    w["qold"] = w["quantity"] * (w["age"] >= old_age)
    agg = w.groupby("project_id")[["quantity", "qa", "qold"]].sum()
    f["vintage_age"] = (agg["qa"] / agg["quantity"]).reindex(f.index)
    f["old_share"] = (agg["qold"] / agg["quantity"]).reindex(f.index)

    # 2) dormancy: quiet period before the CURRENT episode started
    #    (if the episode is the first activity ever: time since listing)
    ep = _episodes(c, f.index, run_gap_days).reindex(f.index)
    listed = projects.set_index("project_id")["listed_at"].reindex(f.index)
    since_listed = (ep["episode_start"] - listed).dt.days
    f["dormancy_days"] = ep["quiet_days"].fillna(since_listed).clip(lower=0)
    f["log_dormancy"] = np.log1p(f["dormancy_days"])
    f["episode_start"] = ep["episode_start"]

    # 3) backlog flush over the whole episode:
    #    retired since episode start / (backlog before episode + issued since start)
    cc = c[c["project_id"].isin(f.index)].merge(
        ep["episode_start"].rename("es"), left_on="project_id", right_index=True)
    before, during = cc[cc["transaction_date"] < cc["es"]], cc[cc["transaction_date"] >= cc["es"]]

    def qsum(df, kind):
        return df[df.transaction_type == kind].groupby("project_id")["quantity"].sum().reindex(f.index).fillna(0)

    backlog = (qsum(before, "issuance") - qsum(before, "retirement")).clip(lower=0)
    f["flush_ratio"] = (qsum(during, "retirement")
                        / (backlog + qsum(during, "issuance")).replace(0, np.nan)).clip(0, 1)

    # 4) burst: window volume vs. the historical average for a window of the same length
    h = past[past["transaction_date"] >= hist_start]
    hist_vol = h.groupby("project_id")["quantity"].sum().reindex(f.index).fillna(0)
    expected = hist_vol / (365 * history_years / window_days)
    f["log_burst"] = np.log((f["win_volume"] + 1) / (expected + 1))

    f = f.fillna({"vintage_age": 0, "old_share": 0, "flush_ratio": 0})
    f["log_volume"] = np.log1p(f["win_volume"])
    # volume of OLD credits moved in the window (not "what moves a lot" but "what moves a lot of OLD credits")
    f["log_old_volume"] = np.log1p(f["win_volume"] * f["old_share"])
    return f


def score(f: pd.DataFrame, weights=WEIGHTS, reason_z=2.0) -> pd.DataFrame:
    if f.empty:
        return f
    explain = list(dict.fromkeys(list(weights) + [k for k in FLAGS if k in f]))
    z = pd.DataFrame({k: _robust_z(f[k]) for k in explain})
    f = f.copy()
    f["score"] = sum(z[k] * w for k, w in weights.items()) / sum(weights.values())
    # tags explain WHY a project is flagged; they do not affect the score
    f["reasons"] = z.apply(
        lambda r: ", ".join(REASONS.get(k, k) for k in explain if r[k] >= reason_z), axis=1
    )
    return f.sort_values("score", ascending=False)


def run(as_of, top=25, **kw):
    credits, projects = load()
    s = score(features(credits, projects, as_of, **kw))
    meta = projects.set_index("project_id")[["name", "registry", "category", "country"]]
    return s.join(meta).head(top)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # avoid encoding errors on Windows consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--window", type=int, default=90)
    a = ap.parse_args()
    out = run(a.as_of, a.top, window_days=a.window)
    cols = ["name", "registry", "score", "vintage_age", "dormancy_days",
            "flush_ratio", "win_volume", "reasons", "bridge_share"]
    pd.set_option("display.width", 200, "display.max_colwidth", 40)
    print(out[cols].round(2).to_string())
