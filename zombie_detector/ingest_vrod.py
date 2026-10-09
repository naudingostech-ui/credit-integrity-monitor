"""
ingest_vrod.py - converts the Berkeley "VROD Registry Files" workbook (raw, transaction-level
registry tables) into the same internal schema that ingest.py produces from OffsetsDB.

Why: an independent second source. If the detector holds up on data collected and processed by a
different team, a processing quirk in OffsetsDB can no longer explain the result.

Source: Berkeley Carbon Trading Project, Voluntary Registry Offsets Database, "Raw registry files".
https://gspp.berkeley.edu/berkeley-carbon-trading-project/offsets-database
The raw files are registry data; keep them and data_vrod/ private (both are git-ignored).

Usage:
    python ingest_vrod.py "data\\VROD-registry-files--2026-08.xlsx"
    set CIM_DATA_DIR=data_vrod   (PowerShell: $env:CIM_DATA_DIR="data_vrod")
    python backtest.py           # now runs on the VROD data

Isometric and ARB/WA sheets are skipped: Isometric is too recent for the 2021-22 backtest, and
ARB/WA are compliance programs with a different layout.
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from ingest import normalize

OUT = Path(__file__).parent / "data_vrod"

# For each sheet: which source column feeds which internal field.
# Names are matched exactly first, then by prefix (some headers are long).
RETIREMENTS = {
    "Verra VCS Retirements": dict(registry="verra", date="Retirement Date", pid="Project ID",
                                  vintage="Vintage", qty="Retirement QTY",
                                  beneficiary="Beneficial Owner", note="Retirement Reason Details",
                                  reason="Retirement Reason", account=None),
    "Gold Retirements": dict(registry="gold-standard", date="Retirement Date", pid="GSID",
                             vintage="Vintage", qty="Quantity", beneficiary="Using Entity",
                             note="Note", reason="Use Case", account=None),
    "ACR Retirements": dict(registry="american-carbon-registry", date="Status Effective (GMT)",
                            pid="Project Id", vintage="Vintage", qty="Quantity of Credits",
                            beneficiary="Retired On Behalf Of", note="Purpose of Retirement",
                            reason="Retirement Reason", account="Credit Holder"),
    "ART Retirements": dict(registry="art-trees", date="Status Effective (GMT)",
                            pid="Project Id", vintage="Vintage", qty="Quantity of Credits",
                            beneficiary="Retired On Behalf Of", note="Purpose of Retirement",
                            reason="Retirement Reason", account="Credit Holder"),
    "CAR Retirements": dict(registry="climate-action-reserve", date="Status Effective",
                            pid="Project ID", vintage="Vintage", qty="Quantity of Offset Credits",
                            beneficiary=None, note="Retirement Reason Details",
                            reason="Retirement Reason", account="Account Holder"),
}
ISSUANCES = {
    "Verra VCS Issuances": dict(registry="verra", date="Issuance Date", pid="Project ID",
                                vintage="Vintage", qty="Units"),
    "Gold Issuances": dict(registry="gold-standard", date="Issuance Date", pid="GSID",
                           vintage="Vintage", qty="Quantity"),
    "ACR Issuances": dict(registry="american-carbon-registry", date="Date Issued (GMT)",
                          pid="Project Id", vintage="Vintage", qty="Total Credits Issued to Project"),
    "ART Issuances": dict(registry="art-trees", date="Date Issued (GMT)", pid="Project Id",
                          vintage="Vintage", qty="Total Credits Issued to Project"),
    "CAR Issuances": dict(registry="climate-action-reserve", date="Date Issued", pid="Project ID",
                          vintage="Vintage", qty="Total Offset Credits Issued"),
}
PROJECTS = {
    "Verra VCS Projects": dict(registry="verra", pid="Project ID", name="Project Name",
                               listed="Project Registration Date", country="Country",
                               category="Sectoral Scope"),
    "Gold Projects": dict(registry="gold-standard", pid="GSID", name="Project Name",
                          listed=None, country="Country", category="Project Type"),
    "ACR Projects": dict(registry="american-carbon-registry", pid="Project Id",
                         name="Project Name", listed="Project Registration/Listing",
                         country=None, category="Project Type"),
    "ART Projects": dict(registry="art-trees", pid="Project Id", name="Project Name",
                         listed="Project Registration/Listing", country=None,
                         category="Project Type"),
    "CAR Projects": dict(registry="climate-action-reserve", pid="Project ID",
                         name="Project Name", listed="Project Listed Date",
                         country="Project Site Country", category="Project Type"),
}
PREFIX = {"verra": "VCS", "gold-standard": "GS"}  # ACR/CAR/ART IDs already carry a prefix


def find(cols, name):
    """Exact header match first, then a unique prefix match. None if absent."""
    if name is None:
        return None
    if name in cols:
        return name
    hits = [c for c in cols if str(c).startswith(name)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous column '{name}': {hits}")
    return None


def project_id(series, registry):
    p = PREFIX.get(registry)
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if p:
        s = s.where(s.str.upper().str.startswith(p), p + s)
    return s


def read_sheet(xl, sheet, spec):
    head = xl.parse(sheet, nrows=0)
    cols = list(head.columns)
    picked = {k: find(cols, v) for k, v in spec.items() if k != "registry"}
    missing = [k for k in ("date", "pid", "qty") if picked.get(k) is None]
    if missing:
        raise ValueError(f"{sheet}: required columns not found {missing}. Headers: {cols}")
    use = [c for c in picked.values() if c is not None]
    df = xl.parse(sheet, usecols=use)
    out = pd.DataFrame({k: (df[c] if c is not None else None) for k, c in picked.items()})
    out["registry"] = spec["registry"]
    return out


def transactions(xl, specs, kind):
    frames = []
    for sheet, spec in specs.items():
        t = read_sheet(xl, sheet, spec)
        f = pd.DataFrame({
            "project_id": project_id(t["pid"], spec["registry"]),
            "quantity": t["qty"],
            "vintage": t.get("vintage"),
            "transaction_date": pd.to_datetime(t["date"], utc=True, errors="coerce", format="mixed"),
            "transaction_type": kind,
            "retirement_account": t.get("account"),
            "retirement_reason": t.get("reason"),
            "retirement_note": t.get("note"),
            "retirement_beneficiary": t.get("beneficiary"),
            "retirement_beneficiary_harmonized": None,
        })
        print(f"  {sheet:<24} {len(f):>9,} rows")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def projects_table(xl):
    frames = []
    for sheet, spec in PROJECTS.items():
        t = read_sheet(xl, sheet, {**spec, "date": spec["pid"], "qty": spec["pid"]})
        frames.append(pd.DataFrame({
            "project_id": project_id(t["pid"], spec["registry"]),
            "name": t["name"], "registry": spec["registry"],
            "country": t.get("country"), "category": t.get("category"),
            "listed_at": t.get("listed"),
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", FutureWarning)  # harmless pandas notice about empty columns
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="VROD-registry-files--YYYY-MM.xlsx")
    a = ap.parse_args()

    print("Opening workbook...")
    xl = pd.ExcelFile(a.xlsx, engine="openpyxl")  # opened once, reused for every sheet
    print("Reading issuances (large sheets take a few minutes)...")
    iss = transactions(xl, ISSUANCES, "issuance")
    print("Reading retirements...")
    ret = transactions(xl, RETIREMENTS, "retirement")
    print("Reading projects...")
    proj = projects_table(xl)

    credits, projects = normalize(pd.concat([iss, ret], ignore_index=True), proj)
    # Registry text fields sometimes hold numbers (e.g. a customer ID in a retirement note).
    # Parquet needs one type per column, so every text column is stored as text.
    for df in (credits, projects):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].where(df[c].isna(), df[c].astype(str))
    OUT.mkdir(exist_ok=True)
    credits.to_parquet(OUT / "credits.parquet", index=False)
    projects.to_parquet(OUT / "projects.parquet", index=False)

    # sanity checks, so a silent mapping error shows up here and not in the backtest
    from detector import BRIDGE_RE
    r = credits[credits.transaction_type == "retirement"]
    text = r[["retirement_beneficiary", "retirement_note", "retirement_account"]] \
        .fillna("").astype(str).agg(" ".join, axis=1)
    bridged = r[text.str.contains(BRIDGE_RE)]
    unknown = set(credits.project_id) - set(projects.project_id)
    print(f"\nOK: {len(projects):,} projects, {len(credits):,} transactions, "
          f"{credits.transaction_date.min():%Y-%m-%d} .. {credits.transaction_date.max():%Y-%m-%d}")
    print(f"Transactions whose project is missing from the project tables: "
          f"{len(set(credits.project_id[credits.project_id.isin(unknown)])):,} projects")
    print(f"Retirements with bridge markers: {len(bridged):,} rows, {bridged.quantity.sum():,} credits")
    print(bridged.assign(y=bridged.transaction_date.dt.year)
          .groupby("y")["quantity"].sum().to_string())


if __name__ == "__main__":
    main()
