"""
Build and cache the datasets used by the ML scripts.

For each world (no trend / with trend) it simulates N_WORKERS workers, computes the
rich features on weeks 0-103, then simulates the loan year (weeks 104-155) under:
  - a fixed repayment schedule and a seasonally adjusted one
  - a normal year, and a stress year: a 12-week income slump (weeks 20-31 of the loan
    year, which is the monsoon dip) of 10% for every worker

Saves data/ml_no_trend.csv and data/ml_with_trend.csv
"""
from pathlib import Path
import sys
import pandas as pd

from simulate_income import simulate_population
from features_ml import rich_features
from loan_sim import simulate_loan

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

OBS_WEEKS = 104
N_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEED = 2024
STRESS_CUT = 0.10                # stress year: income 10% lower ...
STRESS_WEEKS = (20, 32)          # ... for 12 weeks, landing in the monsoon dip
LIVING, LOAN, BUFFER = 0.70, 0.25, 3
WORLDS = {"no_trend": 0.0, "with_trend": 5.0}

for name, trend_std in WORLDS.items():
    inc = simulate_population(n_workers=N_WORKERS, seed=SEED, trend_std=trend_std)
    rows = []
    for wid, g in inc.groupby("worker_id"):
        g = g.sort_values("week")
        obs = g[g.week < OBS_WEEKS]["income"]
        future = g[g.week >= OBS_WEEKS]["income"].to_numpy()
        feats, profile = rich_features(obs)
        stressed = future.copy()
        stressed[STRESS_WEEKS[0]:STRESS_WEEKS[1]] *= 1 - STRESS_CUT
        m = feats["mean_income"]
        row = {"worker_id": wid, "archetype": g["archetype"].iloc[0], **feats}
        for sched, prof in [("fixed", None), ("seasonal", profile)]:
            row[f"default_{sched}"] = simulate_loan(future, m, LIVING, LOAN, BUFFER, prof)
            row[f"default_{sched}_stress"] = simulate_loan(stressed, m, LIVING, LOAN, BUFFER, prof)
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f"  {name}: {len(rows)}/{N_WORKERS}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / f"ml_{name}.csv", index=False)
    print(f"saved data/ml_{name}.csv  shape={df.shape}")
    print(df[["default_fixed", "default_seasonal", "default_fixed_stress",
              "default_seasonal_stress"]].mean().round(3).to_string())