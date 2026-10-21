"""
At a fixed portfolio default target, how many workers get approved under a fixed
vs a seasonally adjusted repayment schedule? Workers are scored out-of-fold,
ranked from lowest to highest predicted risk, and approved while the realised
default rate of the approved group stays at or below TARGET.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from loan_sim import make_dataset, make_labels

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = 0.05
SEEDS = [42, 7, 21]
N_WORKERS = 2000   # larger sample: strict approval targets are noisy with few workers
MODELS = {
    "A baseline": ["mean_income", "raw_cv"],
    "D STL features (no trend)": ["mean_income", "raw_cv", "residual_cv",
                                  "seasonal_amplitude", "max_abs_residual"],
}
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def approve(score, y, target):
    order = np.argsort(score)
    cum_def = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.where(cum_def <= target)[0]
    k = ok.max() + 1 if len(ok) else 0
    mask = np.zeros(len(y), bool)
    mask[order[:k]] = True
    return mask


rows = []
for tname, tstd in {"no trend": 0.0}.items():
    for seed in SEEDS:
        feats, profiles, futures = make_dataset(seed=seed, trend_std=tstd, n_workers=N_WORKERS)
        for schedule in ["fixed", "seasonal"]:
            y = make_labels(feats, profiles, futures, schedule)
            for mname, cols in MODELS.items():
                p = cross_val_predict(clf, feats[cols], y,
                                      cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                      method="predict_proba")[:, 1]
                m = approve(p, y, TARGET)
                r = dict(trend=tname, schedule=schedule, model=mname, seed=seed,
                         approved_all=m.mean())
                for arch in ["steady", "seasonal", "volatile"]:
                    r[f"approved_{arch}"] = m[(feats["archetype"] == arch).values].mean()
                rows.append(r)
        print("done seed", seed, flush=True)

res = pd.DataFrame(rows)
cols = ["approved_all", "approved_steady", "approved_seasonal", "approved_volatile"]
print(f"\n=== Share approved at a {TARGET:.0%} default target (mean over seeds) ===")
print(res.groupby(["trend", "model", "schedule"])[cols].mean().round(3).to_string())
print("\n=== Spread across seeds (std) ===")
print(res.groupby(["trend", "model", "schedule"])[cols].std().round(3).to_string())
res.to_csv(OUT / "policy_sim.csv", index=False)