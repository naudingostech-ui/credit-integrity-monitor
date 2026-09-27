"""
ablation.py - which signals, alone or combined, actually carry information?

v0.2 showed that a sum of signals (AUC ~0.62) did worse than the best single signal
inside it (vintage_age ~0.81), meaning some signals add noise. Here each signal and
several combinations are tested on the same months and the same label.

Candidate added in v0.3: log_old_volume = log(volume of old credits moved in the window).
Idea: not "what moves a lot" (the baseline) but "what moves a lot of OLD credits".

Usage:
    python ablation.py
    python ablation.py --start 2021-11-01 --end 2022-01-01   # time split
    python ablation.py --old-age 4                           # threshold sensitivity
"""
import argparse
import sys

import numpy as np
import pandas as pd

from backtest import at_k, auc
from detector import features, load, score

SUBSETS = {
    "baseline: log_volume": {"log_volume": 1},
    "vintage_age": {"vintage_age": 1},
    "old_share": {"old_share": 1},
    "log_dormancy": {"log_dormancy": 1},
    "flush_ratio": {"flush_ratio": 1},
    "log_burst": {"log_burst": 1},
    "log_old_volume": {"log_old_volume": 1},
    "vintage + old_share": {"vintage_age": 1, "old_share": 1},
    "vintage + old_share + dormancy": {"vintage_age": 1, "old_share": 1, "log_dormancy": 1},
    "vintage + old_share + flush": {"vintage_age": 1, "old_share": 1, "flush_ratio": 1},
    "vintage + log_old_volume": {"vintage_age": 1, "log_old_volume": 1},
    "v0.2 (all 5)": {"vintage_age": 1, "old_share": 1, "log_dormancy": 1, "flush_ratio": 1, "log_burst": 0.3},
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2022-12-01")
    ap.add_argument("--share-min", type=float, default=0.2)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--old-age", type=int, default=6, help="age in years from which a vintage counts as old")
    a = ap.parse_args()

    credits, projects = load()
    months = []
    for m in pd.date_range(a.start, a.end, freq="MS"):
        f = features(credits, projects, m, old_age=a.old_age)
        if f.empty:
            continue
        y = f["bridge_share"] >= a.share_min
        if y.sum() >= a.min_pos:
            months.append((m, f, y))
    if not months:
        print("No eligible months.")
        return

    rows = []
    for name, w in SUBSETS.items():
        aucs, precs = [], []
        for _, f, y in months:
            s = score(f, weights=w)
            aucs.append(auc(s["score"], y.loc[s.index]))
            precs.append(at_k(s["score"], y.loc[s.index], a.k)[0])
        rows.append({"signals": name, "AUC": np.nanmean(aucs), "AUC min": np.nanmin(aucs),
                     f"P@{a.k}": np.mean(precs)})

    df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    base = df.loc[df["signals"] == "baseline: log_volume", "AUC"].iloc[0]
    df["vs baseline"] = df["AUC"] - base
    print(f"Months: {len(months)} ({a.start}..{a.end}), label >= {a.share_min:.0%} via bridge, old = {a.old_age}+ years\n")
    print(df.round(3).to_string(index=False))
    print("\n'AUC min' = worst single month. A robust signal is high on both the mean and the minimum.")


if __name__ == "__main__":
    main()
