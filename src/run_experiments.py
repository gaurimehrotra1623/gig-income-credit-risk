"""
Main hypothesis test: fixed vs seasonally adjusted repayment, across seeds,
with and without trend in the simulator.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from loan_sim import make_dataset, make_labels

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 7, 21, 99, 123]
TREND_SETTINGS = {"with trend (sd=5)": 5.0, "no trend (sd=0)": 0.0}

MODELS = {
    "A baseline (mean, raw CV)": ["mean_income", "raw_cv"],
    "B all STL features": ["mean_income", "raw_cv", "residual_cv", "seasonal_amplitude",
                           "trend_slope", "max_abs_residual"],
    "C mean + residual CV": ["mean_income", "residual_cv"],
    "D all STL except trend": ["mean_income", "raw_cv", "residual_cv",
                               "seasonal_amplitude", "max_abs_residual"],
}

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

rows_rate, rows_auc = [], []
for tname, tstd in TREND_SETTINGS.items():
    for seed in SEEDS:
        feats, profiles, futures = make_dataset(seed=seed, trend_std=tstd)
        for schedule in ["fixed", "seasonal"]:
            y = make_labels(feats, profiles, futures, schedule)
            tmp = feats.assign(default=y)
            for arch, rate in tmp.groupby("archetype")["default"].mean().items():
                rows_rate.append(dict(trend=tname, schedule=schedule, seed=seed,
                                      archetype=arch, default_rate=rate))
            for mname, cols in MODELS.items():
                auc = cross_val_score(clf, feats[cols], y, cv=cv, scoring="roc_auc").mean()
                rows_auc.append(dict(trend=tname, schedule=schedule, seed=seed,
                                     model=mname, auc=auc))
        print(f"done: {tname}, seed {seed}", flush=True)

rate = pd.DataFrame(rows_rate)
auc = pd.DataFrame(rows_auc)

print("\n=== Default rate by worker type (mean over seeds) ===")
print(rate.pivot_table(index=["trend", "schedule"], columns="archetype",
                       values="default_rate", aggfunc="mean").round(3).to_string())

print("\n=== ROC-AUC (mean over seeds) ===")
print(auc.pivot_table(index=["trend", "schedule"], columns="model",
                      values="auc", aggfunc="mean").round(3).to_string())

print("\n=== AUC std across seeds ===")
print(auc.pivot_table(index=["trend", "schedule"], columns="model",
                      values="auc", aggfunc="std").round(3).to_string())

rate.to_csv(OUT / "exp_default_rates.csv", index=False)
auc.to_csv(OUT / "exp_auc.csv", index=False)
print("\nSaved outputs/exp_default_rates.csv and outputs/exp_auc.csv")