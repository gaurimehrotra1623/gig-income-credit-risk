"""
Finance analysis on top of the ML models.

1. Profit curves: profit per applicant as the approval cutoff moves, fixed vs seasonal
   repayment schedule (the cutoff sweep uses outcomes, so it shows the best case).
2. Inclusion: with a cutoff set IN ADVANCE (break-even default rate = margin/(margin+loss)),
   what share of workers who would have repaid does each model approve, by worker type?
   Rejecting someone who would have repaid is the cost of exclusion.
3. Stress test: models are trained on a normal year, then the loan year is replayed with
   a 12-week, 10% income slump that lands in the monsoon dip. How far do realised defaults,
   and profit, move from what the model predicted, and does the repayment schedule change that?

Out-of-fold predictions throughout: no worker is scored by a model that saw them.
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features_ml import BASE, ALL

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MARGIN, LOSS = 0.10, 0.60                  # assumptions, per unit of loan principal
BREAKEVEN = MARGIN / (MARGIN + LOSS)       # approve if predicted default prob <= this
TYPES = ["steady", "seasonal", "volatile"]

MODELS = {
    "Baseline LR": (BASE, make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
    "Gradient boosting": (ALL, HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=6, l2_regularization=1.0, max_iter=150, random_state=0)),
}


def profit(approve, y):
    return float(np.where(approve, (1 - y) * MARGIN - y * LOSS, 0.0).mean())


store = {}   # (world, schedule) -> dict
for world in ["no_trend", "with_trend"]:
    df = pd.read_csv(DATA / f"ml_{world}.csv")
    for sched in ["fixed", "seasonal"]:
        y = df[f"default_{sched}"].values
        preds = {}
        for mname, (cols, est) in MODELS.items():
            preds[mname] = cross_val_predict(
                est, df[cols], y, method="predict_proba",
                cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
        store[(world, sched)] = dict(df=df, y=y, ystress=df[f"default_{sched}_stress"].values,
                                     preds=preds)
    print("fitted", world, flush=True)

# ---------------- 1. profit curves
cut = np.linspace(0.01, 0.60, 120)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
print("\n=== 1. Profit per applicant (gradient boosting) ===")
rows = []
for ax, world in zip(axes, ["no_trend", "with_trend"]):
    for sched, col in [("fixed", "#c0392b"), ("seasonal", "#0f3460")]:
        d = store[(world, sched)]
        p, y = d["preds"]["Gradient boosting"], d["y"]
        appr = np.array([(p <= c).mean() for c in cut])
        prof = np.array([profit(p <= c, y) for c in cut])
        ax.plot(appr, prof, color=col, label=f"{sched} schedule")
        ax.scatter([1.0], [profit(np.ones(len(y), bool), y)], color=col, marker="x", s=60)
        k = prof.argmax()
        ax.scatter([appr[k]], [prof[k]], color=col, s=50, zorder=3)
        rows.append(dict(world=world, schedule=sched,
                         best_profit=prof[k], approval_at_best=appr[k],
                         cutoff_at_best=cut[k],
                         approve_everyone=profit(np.ones(len(y), bool), y),
                         breakeven_rule_profit=profit(p <= BREAKEVEN, y),
                         breakeven_rule_approval=(p <= BREAKEVEN).mean()))
    ax.axhline(0, color="k", lw=0.6)
    ax.set(title=world.replace("_", " "), xlabel="Share of applicants approved")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Profit per applicant (fraction of principal)")
fig.suptitle("Profit vs how many workers the lender approves (dot = best cutoff, x = approve everyone)", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "finance_profit_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------- 2. inclusion at the ex-ante break-even cutoff
print(f"\n=== 2. Approval at the break-even cutoff ({BREAKEVEN:.1%}) by worker type ===")
print("'repaid-approved' = share of workers who would have REPAID that the model approves")
rows = []
for (world, sched), d in store.items():
    arch = d["df"]["archetype"].values
    for mname, p in d["preds"].items():
        ok = p <= BREAKEVEN
        for t in TYPES:
            m = arch == t
            rows.append(dict(world=world, schedule=sched, model=mname, type=t,
                             approved=ok[m].mean(),
                             repaid_approved=ok[m & (d["y"] == 0)].mean() if (m & (d["y"] == 0)).any() else np.nan,
                             default_rate=d["y"][m].mean()))
inc = pd.DataFrame(rows)
inc.to_csv(OUT / "finance_inclusion.csv", index=False)
print(inc.pivot_table(index=["world", "schedule", "model"], columns="type",
                      values="repaid_approved").round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
for r, world in enumerate(["no_trend", "with_trend"]):
    for c, sched in enumerate(["fixed", "seasonal"]):
        ax = axes[r, c]
        g = inc[(inc.world == world) & (inc.schedule == sched)]
        for k, (mname, col) in enumerate([("Baseline LR", "#2a9d8f"), ("Gradient boosting", "#0f3460")]):
            gg = g[g.model == mname].set_index("type").loc[TYPES]
            ax.bar(np.arange(3) + (k - 0.5) * 0.38, gg["repaid_approved"], 0.38, label=mname, color=col)
        ax.set_xticks(range(3))
        ax.set_xticklabels(TYPES)
        ax.set_ylim(0, 1)
        ax.set_title(f"{world.replace('_', ' ')} - {sched} schedule", fontsize=10)
        if c == 0:
            ax.set_ylabel("Share of would-repay workers approved")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Who gets wrongly turned away? (approval among workers who would have repaid)", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "finance_inclusion.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- 3. stress test
print("\n=== 3. Stress test: approvals decided on a normal year, outcomes in a year with a 12-week, 10% income slump ===")
rows = []
for (world, sched), d in store.items():
    p, y, ys = d["preds"]["Gradient boosting"], d["y"], d["ystress"]
    ok = p <= BREAKEVEN
    rows.append(dict(world=world, schedule=sched, approved=ok.mean(),
                     predicted_pd_approved=p[ok].mean(),
                     default_approved_normal=y[ok].mean(),
                     default_approved_stress=ys[ok].mean(),
                     profit_normal=profit(ok, y), profit_stress=profit(ok, ys)))
st = pd.DataFrame(rows)
st.to_csv(OUT / "finance_stress.csv", index=False)
print(st.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, world in zip(axes, ["no_trend", "with_trend"]):
    g = st[st.world == world].set_index("schedule").loc[["fixed", "seasonal"]]
    x = np.arange(2)
    ax.bar(x - 0.27, g["predicted_pd_approved"], 0.27, label="Model's predicted default", color="#999999")
    ax.bar(x, g["default_approved_normal"], 0.27, label="Realised, normal year", color="#0f3460")
    ax.bar(x + 0.27, g["default_approved_stress"], 0.27, label="Realised, stress year", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(["fixed schedule", "seasonal schedule"])
    ax.set_title(world.replace("_", " "))
    ax.set_ylabel("Default rate among approved workers")
axes[0].legend(fontsize=8)
fig.suptitle("Model risk: what a normal-year model misses in a 12-week downturn", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "finance_stress.png", dpi=150, bbox_inches="tight")
print("\nSaved outputs/finance_profit_curves.png, finance_inclusion.png, finance_stress.png (+ csvs)")