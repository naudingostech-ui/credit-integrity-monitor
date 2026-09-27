"""
diagnose.py - why does a detector version lose to the baseline? Three questions:

  A. Which signal carries information on its own (AUC per signal)?
  B. Are the labels right (what BRIDGE_RE actually matches, and what it misses)?
  C. What do the bridged projects the detector missed look like?

Only the bridge peak is analysed (default 2021-11 .. 2022-06), where labels are plentiful.

Usage:
    python diagnose.py > results/diagnose.txt
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd

from backtest import auc
from detector import BRIDGE_RE, WEIGHTS, features, load, score

# Wider net: what BRIDGE_RE might be missing
SUSPECT_RE = re.compile(r"polygon|celo|\bbridge\b(?! \(shanghai)|klimadao|\bc3\b|\bnct\b|\bbct\b|tco2|mco2|\bmoss\b|token|crypto|blockchain|web3", re.I)
ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # avoid encoding errors on Windows consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-11-01")
    ap.add_argument("--end", default="2022-06-01")
    a = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_colwidth", 60, "display.max_rows", 100)

    credits, projects = load()
    frames = []
    for m in pd.date_range(a.start, a.end, freq="MS"):
        s = score(features(credits, projects, m))
        if not s.empty:
            frames.append(s.assign(as_of=m.date()))
    df = pd.concat(frames)
    df["log_volume"] = np.log1p(df["win_volume"])
    y = df["bridge_share"] >= 0.2  # same label as the backtest.py default

    # ---------------- A ----------------
    section("A. AUC per signal (mean over months) and medians")
    feats = list(WEIGHTS) + ["log_volume", "win_retired", "win_issued", "dormancy_days", "score"]
    rows = []
    for f in feats:
        per_month = [auc(g[f], g["bridge_share"] >= 0.2) for _, g in df.groupby("as_of")]
        rows.append({
            "signal": f,
            "AUC": np.nanmean(per_month),
            "median positives": df.loc[y, f].median(),
            "median others": df.loc[~y, f].median(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nAUC < 0.5 means the signal works in the OPPOSITE direction to what we assumed.")

    # ---------------- B ----------------
    section("B. Label audit")
    lo, hi = pd.Timestamp(a.start, tz="UTC") - pd.Timedelta(days=90), pd.Timestamp(a.end, tz="UTC")
    r = credits[(credits.transaction_type == "retirement")
                & credits.transaction_date.between(lo, hi)].copy()
    r["text"] = r[["retirement_beneficiary", "retirement_note", "retirement_account"]] \
        .fillna("").astype(str).agg(" ".join, axis=1)

    def terms(t):
        found = {("0x-address" if ADDR_RE.fullmatch(m) else m.lower()) for m in BRIDGE_RE.findall(t)}
        return ", ".join(sorted(found))

    b = r[r["is_bridge"]].copy()
    b["terms"] = b["text"].apply(terms)
    print("What matched (transactions and credit volume):")
    print(b.groupby("terms")["quantity"].agg(["count", "sum"])
          .sort_values("sum", ascending=False).head(15).to_string())

    print("\n10 random matched records (check by eye that they really are bridges):")
    for t in b["text"].sample(min(10, len(b)), random_state=1):
        print("  +", t.strip()[:150])

    miss = r[~r["is_bridge"] & r["text"].str.contains(SUSPECT_RE)]
    print(f"\nPOSSIBLY MISSED: {len(miss)} transactions, {miss.quantity.sum():,} credits. Examples:")
    for t in miss["text"].drop_duplicates().head(15):
        print("  ?", t.strip()[:150])

    # ---------------- C ----------------
    section("C. Missed positives (label True, but score in the bottom half)")
    df["rank_pct"] = df.groupby("as_of")["score"].rank(pct=True)
    pos = df[y]
    missed, caught = pos[pos.rank_pct < 0.5], pos[pos.rank_pct >= 0.9]
    cols = ["vintage_age", "old_share", "dormancy_days", "flush_ratio", "log_burst", "win_volume"]
    print(f"Caught (top 10%): {len(caught)} | Missed (bottom half): {len(missed)}")
    print("\nMedian comparison:")
    print(pd.DataFrame({"caught": caught[cols].median(), "missed": missed[cols].median(),
                        "negatives": df.loc[~y, cols].median()}).round(2).to_string())

    meta = projects.set_index("project_id")[["name", "category", "country"]]
    print("\n15 largest missed (by volume):")
    top_missed = missed.sort_values("win_volume", ascending=False).drop_duplicates().head(15)
    print(top_missed.join(meta)[["as_of", "name", "category", "country"] + cols].round(2).to_string())


if __name__ == "__main__":
    main()
