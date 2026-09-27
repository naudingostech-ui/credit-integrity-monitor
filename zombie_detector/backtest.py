"""
backtest.py - does the detector find the 2021-22 zombies on its own?

Label: in the window, at least --share-min of a project's retirements carry a bridge
marker (Toucan, C3, Polygon address...). This is a proxy, not ground truth: not every
bridged project is low quality. It does capture the documented 2022 mechanism.

The key comparison is against a naive baseline ("rank by volume"). If the detector
cannot beat it, it adds nothing to simply watching what moves the most.

Usage:
    python backtest.py --start 2021-07-01 --end 2022-12-01
"""
import argparse
import sys

import numpy as np
import pandas as pd

from detector import features, load, score


def auc(scores: pd.Series, labels: pd.Series) -> float:
    """ROC AUC via ranks (Mann-Whitney). 0.5 = random, 1.0 = perfect."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return np.nan
    r = scores.rank()
    return (r[labels].sum() - pos * (pos + 1) / 2) / (pos * neg)


def at_k(scores: pd.Series, labels: pd.Series, k: int):
    top = scores.sort_values(ascending=False).head(k).index
    hits = labels.loc[top].sum()
    return hits / min(k, len(top)), hits / max(labels.sum(), 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # avoid encoding errors on Windows consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2022-12-01")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--label", choices=["share", "any"], default="share",
                    help="share: >= --share-min of retirements via a bridge; any: at least one bridge retirement (v0.1)")
    ap.add_argument("--share-min", type=float, default=0.2)
    ap.add_argument("--min-pos", type=int, default=20, help="skip months with fewer positives (too noisy)")
    a = ap.parse_args()

    credits, projects = load()
    rows = []
    for as_of in pd.date_range(a.start, a.end, freq="MS"):
        s = score(features(credits, projects, as_of, window_days=a.window))
        if s.empty:
            continue
        y = (s["bridge_share"] >= a.share_min) if a.label == "share" else s["label_bridge"].astype(bool)
        if y.sum() < a.min_pos:
            continue
        p_det, r_det = at_k(s["score"], y, a.k)
        p_vol, r_vol = at_k(s["win_volume"], y, a.k)
        rows.append({
            "as_of": as_of.date(), "active": len(s), "positives": int(y.sum()),
            f"P@{a.k}": p_det, f"P@{a.k} base": p_vol,
            f"R@{a.k}": r_det, f"R@{a.k} base": r_vol,
            "AUC": auc(s["score"], y), "AUC base": auc(s["win_volume"], y),
        })

    if not rows:
        print(f"No months with >= {a.min_pos} positives. Try --min-pos 5 or check the dates.")
        return
    df = pd.DataFrame(rows)
    print(f"Label: {a.label}" + (f" (>= {a.share_min:.0%})" if a.label == "share" else "") + f", months with >= {a.min_pos} positives\n")
    print(df.round(2).to_string(index=False))
    print("\nMEAN:")
    print(df.drop(columns=["as_of"]).mean().round(3).to_string())
    lift = df["AUC"].mean() - df["AUC base"].mean()
    print(f"\nDetector vs baseline (AUC): {lift:+.3f} "
          + ("- adds value" if lift > 0.05 else "- does NOT add enough, review the signals"))


if __name__ == "__main__":
    main()
