"""
Step 4 (Phase 3): build the repayment-risk label.

Features come ONLY from weeks 0-103. The label comes ONLY from weeks 104-155,
by simulating a loan taken at week 104. It is NOT derived from the STL residual.

Assumptions (state these in your report):
  - living costs   = LIVING_SHARE * mean income (observation window)
  - weekly payment = LOAN_SHARE   * mean income (observation window)
  - starting cash buffer = BUFFER_WEEKS * weekly payment
  - each week: cash += income - living costs - payment
  - DEFAULT = 1 if cash ever drops below zero during the 52-week loan year
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

OBS_WEEKS = 104
LIVING_SHARE = 0.70
LOAN_SHARE = 0.25
BUFFER_WEEKS = 3


def build_labels(df):
    rows = []
    for wid, g in df.groupby("worker_id"):
        g = g.sort_values("week")
        obs = g[g.week < OBS_WEEKS]["income"]
        future = g[g.week >= OBS_WEEKS]["income"].to_numpy()

        mean_inc = obs.mean()
        living = LIVING_SHARE * mean_inc
        payment = LOAN_SHARE * mean_inc
        cash = BUFFER_WEEKS * payment

        default, weeks_to_default = 0, None
        for i, inc in enumerate(future):
            cash += inc - living - payment
            if cash < 0:
                default, weeks_to_default = 1, i
                break
        rows.append({"worker_id": wid, "default": default,
                     "weeks_to_default": weeks_to_default,
                     "archetype": g["archetype"].iloc[0]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(DATA / "simulated_income.csv")
    labels = build_labels(df)
    labels.to_csv(DATA / "labels.csv", index=False)
    print(f"Overall default rate: {labels['default'].mean():.1%}")
    print(labels.groupby("archetype")["default"].mean().round(3).rename("default_rate"))