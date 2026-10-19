"""
Step 5 (Phase 5): compare credit-risk models.
  A. Baseline   : mean_income + raw_cv
  B. Decomposed : baseline + residual_cv + seasonal_amplitude + trend_slope + max_abs_residual
  C. Residual   : mean_income + residual_cv only
Metric: ROC-AUC, 5-fold stratified CV repeated 5 times.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

feats = pd.read_csv(DATA / "worker_features.csv")
labels = pd.read_csv(DATA / "labels.csv")[["worker_id", "default"]]
df = feats.merge(labels, on="worker_id")

FEATURE_SETS = {
    "A. Baseline (mean + raw CV)": ["mean_income", "raw_cv"],
    "B. Decomposed (baseline + STL features)": ["mean_income", "raw_cv", "residual_cv",
                                               "seasonal_amplitude", "trend_slope", "max_abs_residual"],
    "C. Residual only (mean + residual CV)": ["mean_income", "residual_cv"],
}

y = df["default"].values
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)

print(f"Workers: {len(df)} | default rate: {y.mean():.1%}\n")
print(f"{'Model':45s} {'AUC (mean ± std)':>20s}")
for name, cols in FEATURE_SETS.items():
    scores = cross_val_score(model, df[cols], y, cv=cv, scoring="roc_auc")
    print(f"{name:45s} {scores.mean():.3f} ± {scores.std():.3f}")

plt.figure(figsize=(6, 5))
skf = StratifiedKFold(5, shuffle=True, random_state=0)
for name, cols in FEATURE_SETS.items():
    p = cross_val_predict(model, df[cols], y, cv=skf, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    plt.plot(fpr, tpr, label=f"{name.split('(')[0].strip()} (AUC {roc_auc_score(y, p):.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Default prediction: baseline vs decomposed features")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(OUT / "roc_comparison.png", dpi=150)
print("\nSaved outputs/roc_comparison.png")

cols = FEATURE_SETS["B. Decomposed (baseline + STL features)"]
fit = model.fit(df[cols], y)
coefs = pd.Series(fit[-1].coef_[0], index=cols).sort_values()
print("\nModel B standardized coefficients (positive = raises default risk):")
print(coefs.round(2).to_string())