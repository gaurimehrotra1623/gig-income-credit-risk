"""
Do the conclusions survive different loan assumptions?
For each (living share, payment share, buffer) setting, records:
  gap_fixed / gap_seasonal : seasonal-worker default rate minus steady-worker rate
  cut                      : relative drop in seasonal-worker default from the adjusted schedule
  auc_gain_*               : AUC(D: STL features without trend) minus AUC(A: baseline)
"""
from itertools import product
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

LIVING = [0.65, 0.70, 0.75]
LOAN = [0.20, 0.25, 0.30]
BUFFER = [2, 3, 5]
A = ["mean_income", "raw_cv"]
D = ["mean_income", "raw_cv", "residual_cv", "seasonal_amplitude", "max_abs_residual"]
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)

rows = []
for tname, tstd in {"no trend": 0.0, "with trend": 5.0}.items():
    feats, profiles, futures = make_dataset(seed=42, trend_std=tstd)
    for lv, ln, bf in product(LIVING, LOAN, BUFFER):
        res = {}
        for sched in ["fixed", "seasonal"]:
            y = make_labels(feats, profiles, futures, sched, lv, ln, bf)
            if y.mean() < 0.03 or y.mean() > 0.97:
                res[sched] = None
                continue
            rates = feats.assign(d=y).groupby("archetype")["d"].mean()
            aucA = cross_val_score(clf, feats[A], y, cv=cv, scoring="roc_auc").mean()
            aucD = cross_val_score(clf, feats[D], y, cv=cv, scoring="roc_auc").mean()
            res[sched] = (rates, aucA, aucD, y.mean())
        if res["fixed"] is None or res["seasonal"] is None:
            continue
        rf, aAf, aDf, mf = res["fixed"]
        rs, aAs, aDs, ms = res["seasonal"]
        rows.append(dict(trend=tname, living=lv, loan=ln, buffer=bf,
                         overall_fixed=mf, overall_seasonal=ms,
                         gap_fixed=rf["seasonal"] - rf["steady"],
                         gap_seasonal=rs["seasonal"] - rs["steady"],
                         cut=1 - rs["seasonal"] / rf["seasonal"],
                         auc_gain_fixed=aDf - aAf, auc_gain_seasonal=aDs - aAs))
    print("done", tname, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "sensitivity.csv", index=False)
print(f"\nConfigs kept (default rate between 3% and 97%): {len(df)}")
print(df.groupby("trend")[["gap_fixed", "gap_seasonal", "cut",
                           "auc_gain_fixed", "auc_gain_seasonal"]]
      .agg(["mean", "min", "max"]).round(3).T.to_string())
print("\nShare of configs where STL-features-without-trend beat baseline AUC by > 0.02:")
print((df.assign(f=df.auc_gain_fixed > 0.02, s=df.auc_gain_seasonal > 0.02)
         .groupby("trend")[["f", "s"]].mean().round(2)
         .rename(columns={"f": "fixed schedule", "s": "seasonal schedule"})).to_string())