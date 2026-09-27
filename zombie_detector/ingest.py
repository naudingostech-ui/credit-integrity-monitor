"""
ingest.py - downloads CarbonPlan OffsetsDB and normalises it to the internal schema.

Principle: the detector never reads raw source data. Everything passes through this
layer, so other sources (Verra directly, biodiversity registries, on-chain events)
can be added later without touching the detector.

Usage:
    python ingest.py                     # download the latest snapshot
    python ingest.py --zip path.zip      # use a file you already downloaded
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://carbonplan-offsets-db.s3.us-west-2.amazonaws.com/production/latest/offsets-db.parquet.zip"
DATA = Path(__file__).parent / "data"

CREDIT_COLS = [
    "project_id", "quantity", "vintage", "transaction_date", "transaction_type",
    "retirement_account", "retirement_reason", "retirement_note",
    "retirement_beneficiary", "retirement_beneficiary_harmonized",
]
PROJECT_COLS = [
    "project_id", "name", "registry", "proponent", "status", "country",
    "category", "project_type", "protocol", "listed_at", "first_issuance_at",
    "issued", "retired",
]


def _read_any(zf: zipfile.ZipFile, keyword: str) -> pd.DataFrame:
    """Finds the file whose name contains the keyword (credit / project) and reads it."""
    names = [n for n in zf.namelist()
             if keyword in n.lower() and n.lower().endswith((".parquet", ".csv"))]
    if not names:
        raise FileNotFoundError(f"No file containing '{keyword}' in the zip: {zf.namelist()}")
    name = sorted(names, key=len)[0]
    raw = zf.read(name)
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def normalize(credits: pd.DataFrame, projects: pd.DataFrame):
    credits = credits.reindex(columns=CREDIT_COLS)
    projects = projects.reindex(columns=PROJECT_COLS)

    credits["transaction_date"] = pd.to_datetime(credits["transaction_date"], utc=True, errors="coerce")
    credits["quantity"] = pd.to_numeric(credits["quantity"], errors="coerce").fillna(0).astype("int64")
    credits["vintage"] = pd.to_numeric(credits["vintage"], errors="coerce").astype("Int64")
    credits["transaction_type"] = credits["transaction_type"].str.lower().str.strip()
    credits = credits.dropna(subset=["transaction_date", "project_id"])
    credits = credits[credits["transaction_type"].isin(["issuance", "retirement"])]

    for col in ["listed_at", "first_issuance_at"]:
        projects[col] = pd.to_datetime(projects[col], utc=True, errors="coerce")
    # protocol is a list; the detector only needs text
    projects["protocol"] = projects["protocol"].apply(
        lambda p: ",".join(map(str, p)) if isinstance(p, (list, tuple)) or hasattr(p, "tolist") else p
    )
    return credits.reset_index(drop=True), projects.drop_duplicates("project_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="local offsets-db.parquet.zip (instead of downloading)")
    args = ap.parse_args()

    if args.zip:
        blob = Path(args.zip).read_bytes()
    else:
        print(f"Downloading {URL} ...")
        r = requests.get(URL, timeout=300)
        r.raise_for_status()
        blob = r.content

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        credits = _read_any(zf, "credit")
        projects = _read_any(zf, "project")

    credits, projects = normalize(credits, projects)
    DATA.mkdir(exist_ok=True)
    credits.to_parquet(DATA / "credits.parquet", index=False)
    projects.to_parquet(DATA / "projects.parquet", index=False)

    last = credits["transaction_date"].max()
    print(f"OK: {len(projects):,} projects, {len(credits):,} transactions, latest {last:%Y-%m-%d}")
    if (pd.Timestamp.now(tz="UTC") - last).days > 30:
        print("WARNING: data is older than 30 days. Fine for backtesting, not for live monitoring.")


if __name__ == "__main__":
    main()
