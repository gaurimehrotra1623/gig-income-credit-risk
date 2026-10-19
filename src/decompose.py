"""
Step 2: STL decomposition + feature engineering.
Features use ONLY the first 104 weeks; later weeks are held out for the loan outcome.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OBS_WEEKS = 104


def worker_features(income, period=52):
    income = income.reset_index(drop=True)
    mean, std = income.mean(), income.std()
    f = {"mean_income": mean, "raw_std": std, "raw_cv": std / mean}

    if len(income) < 2 * period:      # STL needs ~2 full seasonal cycles
        f.update(dict.fromkeys(["trend_slope", "seasonal_amplitude",
                                "residual_std", "residual_cv",
                                "max_abs_residual"], np.nan))
        return f

    res = STL(income, period=period, robust=True).fit()
    f.update({
        "trend_slope": np.polyfit(np.arange(len(res.trend)), res.trend, 1)[0],
        "seasonal_amplitude": res.seasonal.max() - res.seasonal.min(),
        "residual_std": res.resid.std(),
        "residual_cv": res.resid.std() / mean,
        "max_abs_residual": res.resid.abs().max(),
    })
    return f


def build_feature_table(df, period=52, obs_weeks=OBS_WEEKS):
    df = df[df["week"] < obs_weeks]
    rows = []
    for wid, g in df.groupby("worker_id"):
        g = g.sort_values("week")
        feats = worker_features(g["income"], period)
        feats["worker_id"] = wid
        feats["archetype"] = g["archetype"].iloc[0]
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(DATA / "simulated_income.csv")
    feats = build_feature_table(df)
    feats.to_csv(DATA / "worker_features.csv", index=False)
    print(feats.groupby("archetype")[["raw_cv", "residual_cv", "seasonal_amplitude"]].mean().round(3))