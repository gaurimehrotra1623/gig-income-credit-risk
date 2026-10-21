"""
Lender economics, with an approval rule that does NOT peek at outcomes.

Rule: approve a worker if the model's out-of-fold predicted default probability is at or
below the break-even default rate, margin / (margin + loss).

Per approved loan (in units of loan principal):
  performing loan -> +margin        defaulted loan -> -loss
Profit per applicant = average over ALL applicants (rejected ones earn 0).

Assumptions (not estimated from data): margin and loss values below, no cost of running a
seasonal schedule, same total interest collected under both schedules.
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

SEEDS = [42, 7, 21]
N_WORKERS = 500
SCENARIOS = {"margin 10%, loss 60%": (0.10, 0.60),
             "margin 15%, loss 60%": (0.15, 0.60),
             "margin 10%, loss 90%": (0.10, 0.90)}
MODELS = {
    "A baseline": ["mean_income", "raw_cv"],
    "D STL features (no trend)": ["mean_income", "raw_cv", "residual_cv",
                                  "seasonal_amplitude", "max_abs_residual"],
}
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

rows = []
for tname, tstd in {"no trend": 0.0, "with trend": 5.0}.items():
    for seed in SEEDS:
        feats, profiles, futures = make_dataset(seed=seed, trend_std=tstd, n_workers=N_WORKERS)
        arch = feats["archetype"].values
        for schedule in ["fixed", "seasonal"]:
            y = make_labels(feats, profiles, futures, schedule)
            preds = {"approve everyone": np.ones(len(y))}
            for mname, cols in MODELS.items():
                preds[mname] = cross_val_predict(
                    clf, feats[cols], y, method="predict_proba",
                    cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
            for mname, p in preds.items():
                for sname, (margin, loss) in SCENARIOS.items():
                    cutoff = margin / (margin + loss)
                    ok = np.ones(len(y), bool) if mname == "approve everyone" else (p <= cutoff)
                    profit = np.where(ok, (1 - y) * margin - y * loss, 0.0)
                    rows.append(dict(
                        trend=tname, scenario=sname, schedule=schedule, model=mname, seed=seed,
                        approval=ok.mean(),
                        approved_default=y[ok].mean() if ok.any() else np.nan,
                        profit_per_applicant=profit.mean(),
                        approved_seasonal=ok[arch == "seasonal"].mean()))
        print("done", tname, seed, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "lender_economics.csv", index=False)
cols = ["approval", "approved_seasonal", "approved_default", "profit_per_applicant"]
g = res.groupby(["trend", "scenario", "model", "schedule"])
print("\n=== Mean over seeds ===")
print(g[cols].mean().round(3).to_string())
print("\n=== Std of profit_per_applicant across seeds ===")
print(g["profit_per_applicant"].std().round(3).to_string())