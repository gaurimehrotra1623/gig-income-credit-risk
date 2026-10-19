"""
Step 1: simulate weekly income for gig workers.

income = base + trend + seasonal + noise + shocks
Three archetypes:
  steady   - low seasonality, low shock risk
  seasonal - high seasonality, low shock risk (predictable volatility)
  volatile - moderate seasonality, high shock risk (genuine risk)

NOTE: this is SIMULATED data, calibrated loosely to published aggregate
stats (JPMorgan Chase Institute, Fairwork India, NITI Aayog).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def simulate_worker(worker_id, n_weeks=104, base_income=6000.0, trend_per_week=0.0,
                    seasonal_amplitude=1200.0, noise_std=400.0, shock_prob=0.04,
                    rng=None):
    rng = rng or np.random.default_rng()
    weeks = np.arange(n_weeks)

    trend = trend_per_week * weeks

    annual = np.sin(2 * np.pi * (weeks - 13) / 52)
    festival = 0.6 * np.exp(-0.5 * ((weeks % 52 - 43) / 4) ** 2)   # Oct-Nov bump
    monsoon = -0.5 * np.exp(-0.5 * ((weeks % 52 - 26) / 5) ** 2)   # Jul-Aug dip
    seasonal = seasonal_amplitude * (0.4 * annual + festival + monsoon)

    noise = rng.normal(0, noise_std, n_weeks)

    shocks = np.zeros(n_weeks)
    hit = rng.random(n_weeks) < shock_prob
    shocks[hit] = -np.clip(rng.normal(2500, 1200, hit.sum()), 0, None)

    income = np.clip(base_income + trend + seasonal + noise + shocks, 0, None)
    return pd.DataFrame({"worker_id": worker_id, "week": weeks,
                         "income": income, "had_shock": hit.astype(int)})


def simulate_population(n_workers=500, n_weeks=156, seed=42):
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["steady", "seasonal", "volatile"], size=n_workers,
                       p=[0.35, 0.35, 0.30])
    params = {"steady": (400, 300, 0.015),
              "seasonal": (1800, 350, 0.015),
              "volatile": (900, 400, 0.08)}
    frames = []
    for i, kind in enumerate(kinds):
        amp, noise, shock_p = params[kind]
        df = simulate_worker(i, n_weeks, base_income=rng.normal(6000, 800),
                             trend_per_week=rng.normal(0, 5),
                             seasonal_amplitude=amp, noise_std=noise,
                             shock_prob=shock_p, rng=rng)
        df["archetype"] = kind
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    pop = simulate_population()
    pop.to_csv(DATA / "simulated_income.csv", index=False)
    print(f"Saved {pop.worker_id.nunique()} workers, {len(pop)} rows -> data/simulated_income.csv")
