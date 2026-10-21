"""
Shared loan-simulation helpers for the advanced experiments.

Two repayment schedules, same total repaid in expectation:
  fixed    : the same payment every week
  seasonal : payment follows the worker's own STL seasonal pattern
"""
import numpy as np
from statsmodels.tsa.seasonal import STL

from simulate_income import simulate_population
from decompose import build_feature_table, OBS_WEEKS


def seasonal_profile(obs_income, period=52):
    """Last full seasonal cycle from STL on the observation window, zero-centred.
    Index i lines up with loan week i, because OBS_WEEKS is a multiple of the period."""
    res = STL(obs_income.reset_index(drop=True), period=period, robust=True).fit()
    prof = res.seasonal.to_numpy()[-period:]
    return prof - prof.mean()


def simulate_loan(future, mean_inc, living_share, loan_share, buffer_weeks, profile=None):
    """Return 1 if the worker's cash ever goes below zero during the loan year."""
    living = living_share * mean_inc
    base_pay = loan_share * mean_inc
    cash = buffer_weeks * base_pay
    for i, inc in enumerate(future):
        if profile is None:
            pay = base_pay
        else:
            pay = base_pay * max(0.0, 1 + profile[i % len(profile)] / mean_inc)
        cash += inc - living - pay
        if cash < 0:
            return 1
    return 0


def make_dataset(seed=42, trend_std=5.0, n_workers=500):
    """Simulate workers, build STL features on weeks 0-103, keep what the labels need."""
    inc = simulate_population(n_workers=n_workers, seed=seed, trend_std=trend_std)
    feats = build_feature_table(inc)
    profiles, futures = {}, {}
    for wid, g in inc.groupby("worker_id"):
        g = g.sort_values("week")
        profiles[wid] = seasonal_profile(g[g.week < OBS_WEEKS]["income"])
        futures[wid] = g[g.week >= OBS_WEEKS]["income"].to_numpy()
    return feats, profiles, futures


def make_labels(feats, profiles, futures, schedule="fixed",
                living=0.70, loan=0.25, buffer=3):
    out = []
    for wid, m in zip(feats["worker_id"], feats["mean_income"]):
        prof = profiles[wid] if schedule == "seasonal" else None
        out.append(simulate_loan(futures[wid], m, living, loan, buffer, prof))
    return np.array(out)