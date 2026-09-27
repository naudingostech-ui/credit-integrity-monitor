"""
synth.py - synthetic data for testing the pipeline offline.

Creates three kinds of project:
  - normal: steady issuance and retirement of recent vintages
  - decoys: large, healthy projects with high recent volume, to check that the
    detector is not fooled by volume alone
  - zombies: old vintages, long dormancy, bulk retirement to a 0x address in 2021-10..2022-05

NOTE: this only tests the code. The detector's real value is shown only by the
backtest on OffsetsDB.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DATA = Path(__file__).parent / "data"
rows, projects = [], []


def tx(pid, date, qty, vintage, kind, benef=None):
    rows.append({"project_id": pid, "quantity": int(qty), "vintage": vintage,
                 "transaction_date": pd.Timestamp(date, tz="UTC"), "transaction_type": kind,
                 "retirement_beneficiary": benef, "retirement_note": None,
                 "retirement_account": None, "retirement_reason": None,
                 "retirement_beneficiary_harmonized": None})


def proj(pid, name, listed):
    projects.append({"project_id": pid, "name": name, "registry": "verra",
                     "proponent": None, "status": "registered", "country": "X",
                     "category": "forest", "project_type": "x", "protocol": "vm0000",
                     "listed_at": pd.Timestamp(listed, tz="UTC"),
                     "first_issuance_at": None, "issued": 0, "retired": 0})


def steady(pid, scale, start_year=2014):
    for y in range(start_year, 2024):
        for m in (2, 5, 8, 11):
            d = pd.Timestamp(y, m, rng.integers(1, 28))
            q = scale * rng.uniform(0.5, 1.5)
            tx(pid, d, q, y - 1, "issuance")
            tx(pid, d + pd.Timedelta(days=int(rng.integers(20, 200))), q * 0.8, y - 1,
               "retirement", "Some Corp")


for i in range(300):
    proj(f"N{i}", f"Normal {i}", "2013-01-01")
    steady(f"N{i}", rng.lognormal(8, 1))

for i in range(15):
    proj(f"D{i}", f"Decoy {i}", "2018-01-01")
    steady(f"D{i}", rng.lognormal(12, 0.3), 2019)

for i in range(25):
    pid = f"Z{i}"
    proj(pid, f"Zombie {i}", "2010-06-01")
    v = int(rng.integers(2008, 2013))
    stock = rng.lognormal(10.5, 0.8)
    tx(pid, f"{v + 2}-03-01", stock, v, "issuance")
    tx(pid, f"{v + 2}-09-01", stock * 0.05, v, "retirement", "Old buyer")
    t = pd.Timestamp("2021-10-15") + pd.Timedelta(days=int(rng.integers(0, 200)))
    tx(pid, t, stock * rng.uniform(0.6, 0.95), v, "retirement",
       "Toucan 0x" + "".join(rng.choice(list("0123456789abcdef"), 40)))

DATA.mkdir(exist_ok=True)
pd.DataFrame(rows).to_parquet(DATA / "credits.parquet", index=False)
pd.DataFrame(projects).to_parquet(DATA / "projects.parquet", index=False)
print(f"Synthetic data: {len(projects)} projects, {len(rows)} transactions -> {DATA}")
