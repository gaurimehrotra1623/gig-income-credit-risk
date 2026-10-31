"""
Feature-family ablation: the core critical test of the project.

Does the STL decomposition give a lender anything it could not already get from
plain cash-flow statistics (income floor, worst 4/12-week stretch, downside deviation)?

Feature sets (all on the same workers, same repeated cross-validation):
  Baseline               mean income, raw CV
  Residual only          mean income, residual CV     (my original hypothesis)
  + cash-flow stats      baseline + cash-flow family
  + cash-flow + simple trend   everything computable WITHOUT a decomposition
  + cash-flow + STL      baseline + cash-flow + STL family
  All features           everything

Two learners: logistic regression and gradient boosting.
Runs all four combinations of world (no trend / with trend) and schedule (fixed / seasonal).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from features_ml import BASE, CASHFLOW, TREND_SIMPLE, STLF

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SETS = {
    "Baseline": BASE,
    "Residual only": ["mean_income", "residual_cv"],
    "+ cash-flow stats": BASE + CASHFLOW,
    "+ cash-flow + simple trend": BASE + CASHFLOW + TREND_SIMPLE,
    "+ cash-flow + STL": BASE + CASHFLOW + STLF,
    "All features": BASE + CASHFLOW + TREND_SIMPLE + STLF,
}
LEARNERS = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "Gradient boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=6, l2_regularization=1.0, max_iter=150, random_state=0),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)

rows = []
for world in ["no_trend", "with_trend"]:
    df = pd.read_csv(DATA / f"ml_{world}.csv")
    for schedule in ["fixed", "seasonal"]:
        y = df[f"default_{schedule}"].values
        for sname, cols in SETS.items():
            for lname, est in LEARNERS.items():
                s = cross_val_score(est, df[cols], y, cv=cv, scoring="roc_auc")
                rows.append(dict(world=world, schedule=schedule, features=sname,
                                 learner=lname, auc=s.mean(), sd=s.std()))
        print("done", world, schedule, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "ablation_results.csv", index=False)

for (world, schedule), g in res.groupby(["world", "schedule"]):
    print(f"\n=== {world}, {schedule} schedule (CV AUC, mean +/- sd over folds) ===")
    t = g.pivot(index="features", columns="learner", values="auc").loc[list(SETS)]
    s = g.pivot(index="features", columns="learner", values="sd").loc[list(SETS)]
    print(t.round(3).astype(str).add(" +/- ").add(s.round(3).astype(str)).to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharey=True)
x = np.arange(len(SETS))
for ax, ((world, schedule), g) in zip(axes.ravel(), res.groupby(["world", "schedule"])):
    for k, (lname, col) in enumerate([("Logistic regression", "#2a9d8f"), ("Gradient boosting", "#0f3460")]):
        gg = g[g.learner == lname].set_index("features").loc[list(SETS)]
        ax.bar(x + (k - 0.5) * 0.38, gg["auc"], 0.38, yerr=gg["sd"], capsize=2, label=lname, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(" + ", "\n+ ") for k in SETS], fontsize=7)
    ax.set_title(f"{world.replace('_', ' ')} - {schedule} schedule", fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("CV ROC-AUC")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Which feature families carry default-risk information?", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "ablation_families.png", dpi=150, bbox_inches="tight")
print("\nSaved outputs/ablation_families.png and ablation_results.csv")