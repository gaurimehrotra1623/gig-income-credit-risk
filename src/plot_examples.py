"""
Step 3: plot STL decomposition for one worker of each archetype.
Saves PNGs to outputs/ (use these in your report / LinkedIn post).
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "simulated_income.csv")

for kind in ["steady", "seasonal", "volatile"]:
    wid = df[df.archetype == kind].worker_id.iloc[0]
    s = df[(df.worker_id == wid) & (df.week < 104)].sort_values("week")["income"].reset_index(drop=True)
    fig = STL(s, period=52, robust=True).fit().plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(f"STL decomposition - {kind} worker", y=1.0)
    fig.savefig(OUT / f"stl_{kind}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", f"outputs/stl_{kind}.png")
