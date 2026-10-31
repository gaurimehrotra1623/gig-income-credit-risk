"""
Model comparison on the cached datasets.

Usage:  python src/ml_models.py [world] [schedule]
        world    = no_trend | with_trend   (default no_trend)
        schedule = fixed | seasonal        (default fixed)

Models: baseline logistic regression, logistic regression (L2 and L1) on all features,
random forest, gradient boosting, and an "oracle" that is also told each worker's true
type (an upper bound: how much risk information is simply not visible in the data).

Metrics (standard in credit risk): ROC-AUC, Gini = 2*AUC - 1, KS statistic, Brier score,
average precision. 95% bootstrap intervals on the held-out test set, and a paired
bootstrap test of each model against the baseline.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from features_ml import BASE, ALL

# scikit-learn 1.8 warns about the 'penalty' argument; harmless, so keep the output readable
warnings.filterwarnings("ignore", message=r".*(penalty|l1_ratio).*")

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

world = sys.argv[1] if len(sys.argv) > 1 else "no_trend"
schedule = sys.argv[2] if len(sys.argv) > 2 else "fixed"
tag = f"{world}_{schedule}"

df = pd.read_csv(DATA / f"ml_{world}.csv")
y = df[f"default_{schedule}"].values
X = df[ALL].copy()
X_oracle = pd.concat([X, pd.get_dummies(df["archetype"], prefix="type").astype(float)], axis=1)

idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.30,
                                       stratify=y, random_state=0)
print(f"world={world} schedule={schedule} | workers={len(df)} | default rate={y.mean():.1%}")
print(f"train={len(idx_train)} test={len(idx_test)}\n")

lr = lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **kw))
SPECS = {
    "Baseline LR (mean, raw CV)": (X[BASE], lr(),
                                   {}),
    "LR, all features": (X, lr(),
                         {"logisticregression__C": [0.1, 1, 10]}),
    "LR (L1), all features": (X, lr(penalty="l1", solver="liblinear"),
                              {"logisticregression__C": [0.05, 0.2, 1]}),
    "Random forest": (X, RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                      {"min_samples_leaf": [5, 20]}),
    "Gradient boosting": (X, HistGradientBoostingClassifier(random_state=0),
                          {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [4, 8],
                           "l2_regularization": [0.0, 1.0]}),
    "Oracle (also knows worker type)": (X_oracle, HistGradientBoostingClassifier(
                                            random_state=0, learning_rate=0.05, max_leaf_nodes=6),
                                        {}),
}

cv = StratifiedKFold(5, shuffle=True, random_state=0)
fitted, proba = {}, {}
for name, (Xm, est, grid) in SPECS.items():
    Xtr, Xte = Xm.iloc[idx_train], Xm.iloc[idx_test]
    if grid:
        est = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=1).fit(Xtr, y[idx_train])
        est = est.best_estimator_
    else:
        est = est.fit(Xtr, y[idx_train])
    fitted[name] = (est, Xte)
    proba[name] = est.predict_proba(Xte)[:, 1]
    print("fitted:", name, flush=True)

yt = y[idx_test]


def ks_stat(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


rng = np.random.default_rng(0)
B = 500
boot = {n: [] for n in proba}
for _ in range(B):
    i = rng.integers(0, len(yt), len(yt))
    if yt[i].min() == yt[i].max():
        continue
    for n, p in proba.items():
        boot[n].append(roc_auc_score(yt[i], p[i]))
boot = {n: np.array(v) for n, v in boot.items()}

base = "Baseline LR (mean, raw CV)"
rows = []
for n, p in proba.items():
    auc = roc_auc_score(yt, p)
    lo, hi = np.percentile(boot[n], [2.5, 97.5])
    diff = boot[n] - boot[base]
    rows.append({"model": n, "AUC": auc, "AUC 95% CI": f"{lo:.3f}-{hi:.3f}",
                 "Gini": 2 * auc - 1, "KS": ks_stat(yt, p),
                 "Brier": brier_score_loss(yt, p), "AvgPrec": average_precision_score(yt, p),
                 "AUC gain vs baseline": auc - roc_auc_score(yt, proba[base]),
                 "P(gain<=0)": float((diff <= 0).mean()) if n != base else np.nan})
metrics = pd.DataFrame(rows).set_index("model")
print("\n=== Test-set metrics ===")
print(metrics.round(3).to_string())
metrics.to_csv(OUT / f"ml_metrics_{tag}.csv")

# ---- Figure 1: ROC + calibration + AUC bars
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for n, p in proba.items():
    fpr, tpr, _ = roc_curve(yt, p)
    ax[0].plot(fpr, tpr, label=f"{n} ({roc_auc_score(yt, p):.3f})",
               ls="--" if "Oracle" in n else "-")
ax[0].plot([0, 1], [0, 1], "k:", alpha=0.4)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
ax[0].legend(fontsize=7, loc="lower right")

for n in ["Baseline LR (mean, raw CV)", "LR, all features", "Gradient boosting"]:
    frac, mean_p = calibration_curve(yt, proba[n], n_bins=8, strategy="quantile")
    ax[1].plot(mean_p, frac, marker="o", label=n)
lim = max(0.05, float(max(proba[n].max() for n in proba)))
ax[1].plot([0, lim], [0, lim], "k:", alpha=0.4)
ax[1].set(xlabel="Predicted default probability", ylabel="Observed default rate",
          title="Calibration (closer to the dotted line = better)")
ax[1].legend(fontsize=8)

names = list(proba)
aucs = [roc_auc_score(yt, proba[n]) for n in names]
errs = [[a - np.percentile(boot[n], 2.5) for a, n in zip(aucs, names)],
        [np.percentile(boot[n], 97.5) - a for a, n in zip(aucs, names)]]
ax[2].barh(range(len(names)), aucs, xerr=errs, color="#0f3460", alpha=0.85, capsize=3)
ax[2].set_yticks(range(len(names)))
ax[2].set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
ax[2].set(xlabel="Test AUC with 95% bootstrap interval", title="Do the models really differ?")
ax[2].set_xlim(max(0.4, min(aucs) - 0.1), 1.0)
ax[2].invert_yaxis()
fig.suptitle(f"Model comparison - {world.replace('_', ' ')}, {schedule} repayment schedule", y=1.02)
fig.tight_layout()
fig.savefig(OUT / f"ml_compare_{tag}.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: permutation importance for gradient boosting (AUC drop when a feature is shuffled)
est, Xte = fitted["Gradient boosting"]
pi = permutation_importance(est, Xte, yt, scoring="roc_auc", n_repeats=15, random_state=0)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values()
err = pd.Series(pi.importances_std, index=Xte.columns)[imp.index]
from features_ml import CASHFLOW, STLF, TREND_SIMPLE
colors = ["#e07a1f" if c in STLF else "#2a9d8f" if c in CASHFLOW + TREND_SIMPLE else "#0f3460" for c in imp.index]
top = imp.tail(12)
plt.figure(figsize=(7, 5))
plt.barh(top.index, top.values, xerr=err[top.index].values, color=colors[-12:], capsize=2)
plt.xlabel("Drop in test AUC when the feature is shuffled")
plt.title(f"What the gradient boosting model relies on\n(navy = baseline, teal = cash-flow/simple trend, orange = STL) - {tag}", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / f"ml_importance_{tag}.png", dpi=150)
plt.close()
print("\nTop features by permutation importance:")
print(imp.tail(8)[::-1].round(4).to_string())
print(f"\nSaved outputs/ml_compare_{tag}.png, ml_importance_{tag}.png, ml_metrics_{tag}.csv")