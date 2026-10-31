"""
Richer feature set for the ML experiments, in three families:

  BASE      what a naive lender would use
  CASHFLOW  direct cash-flow statistics a lender could compute WITHOUT any decomposition
  TREND_SIMPLE  a trend feature from a plain straight-line fit on raw income (no decomposition)
  STLF      features that need the STL decomposition (trend / seasonal / residual)

The central questions for the ML work: do the STL features add anything beyond
the cash-flow statistics, and is the decomposition needed at all for detecting a trend?
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

PERIOD = 52

BASE = ["mean_income", "raw_cv"]
CASHFLOW = ["p10_ratio", "low_week_share", "downside_dev",
            "worst_4wk_ratio", "worst_12wk_ratio", "raw_skew"]
TREND_SIMPLE = ["ols_slope_ratio"]
STLF = ["trend_slope", "trend_end_ratio", "seasonal_amplitude", "seasonal_strength",
        "trend_strength", "residual_cv", "max_abs_residual", "resid_skew",
        "shock_count", "resid_acf1"]
ALL = BASE + CASHFLOW + TREND_SIMPLE + STLF


def rich_features(income, period=PERIOD):
    """Return (feature dict, zero-centred seasonal profile of the last cycle)."""
    x = pd.Series(np.asarray(income, dtype=float))
    mean, std = x.mean(), x.std()
    f = {"mean_income": mean, "raw_std": std, "raw_cv": std / mean}

    # cash-flow family (no decomposition needed)
    f["p10_ratio"] = x.quantile(0.10) / mean
    f["low_week_share"] = float((x < 0.7 * mean).mean())
    f["downside_dev"] = float(np.sqrt(np.mean(np.minimum(x - mean, 0) ** 2)) / mean)
    f["worst_4wk_ratio"] = x.rolling(4).mean().min() / mean
    f["worst_12wk_ratio"] = x.rolling(12).mean().min() / mean
    f["raw_skew"] = float(stats.skew(x))

    # simple trend: straight-line fit on raw income, as a fraction of mean income over the window
    ols = np.polyfit(np.arange(len(x)), x.to_numpy(), 1)[0]
    f["ols_slope_ratio"] = ols * len(x) / mean

    # STL family
    res = STL(x, period=period, robust=True).fit()
    trend, seas, resid = res.trend, res.seasonal, res.resid
    f["trend_slope"] = np.polyfit(np.arange(len(trend)), trend, 1)[0]
    f["trend_end_ratio"] = trend.iloc[-1] / mean          # where income is heading vs its average
    f["seasonal_amplitude"] = seas.max() - seas.min()
    f["seasonal_strength"] = max(0.0, 1 - resid.var() / (resid + seas).var())
    f["trend_strength"] = max(0.0, 1 - resid.var() / (resid + trend).var())
    f["residual_std"] = resid.std()
    f["residual_cv"] = resid.std() / mean
    f["max_abs_residual"] = resid.abs().max()
    f["resid_skew"] = float(stats.skew(resid))
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    f["shock_count"] = int((resid < -2.5 * mad).sum())    # unusually bad weeks
    f["resid_acf1"] = float(acf(resid, nlags=1)[1])       # do bad weeks cluster?

    profile = seas.to_numpy()[-period:]
    return f, profile - profile.mean()