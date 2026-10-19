# Gig income volatility and credit risk

I'm testing one idea: that credit models treat all income volatility as risk, when for gig workers a lot of it is predictable (festival weeks, monsoon dips, weekly demand cycles). If that's true, splitting income into a predictable part and a leftover unexplained part should give a better risk signal than average income or raw variance.

This is an independent project, built alongside reading for my MBA applications. It uses simulated data, and I say why below.

## The question

Do features from an STL decomposition of weekly income, especially the residual volatility, predict loan default better than a baseline built on mean income and raw volatility?

## Pipeline

```
simulate_income.py   156 weeks of income for 500 simulated workers
        |
        |-- weeks 0-103   -->  decompose.py   -->  worker_features.csv
        |                      (STL, period = 52)
        |
        '-- weeks 104-155 -->  risk_proxy.py  -->  labels.csv
                               (simulated loan, did the worker default?)

worker_features.csv + labels.csv  -->  train_model.py  -->  AUC comparison, ROC plot
```

The split in the middle is deliberate. Features only see the first two years of income. The default label only comes from the third year. I did this so the label is not computed from the same residual I'm using as a feature, which would make the comparison circular.

## What each script does

**simulate_income.py.** Builds weekly income as base level + slow trend + seasonal pattern (a festival bump around week 43, a monsoon dip around week 26) + noise + occasional income shocks. Workers come in three types:

| Type | Share | Seasonal swing | Shock chance per week |
|---|---|---|---|
| steady | 35% | small | 1.5% |
| seasonal | 35% | large | 1.5% |
| volatile | 30% | medium | 8% |

**decompose.py.** Runs robust STL on each worker's first 104 weeks and pulls out: mean income, raw std and raw CV (the baseline), plus trend slope, seasonal amplitude, residual std, residual CV and the largest single residual. STL needs about two full seasonal cycles, which is why the observation window is 104 weeks.

**risk_proxy.py.** Simulates a loan taken at week 104. Weekly living costs are 70% of the worker's mean income, the weekly payment is 25%, and they start with a cash buffer of 3 payments. Each week cash changes by income minus living costs minus the payment. If cash goes below zero at any point in the 52 weeks, that worker defaults.

**train_model.py.** Logistic regression on standardised features, scored by ROC-AUC with 5-fold cross-validation repeated 5 times. Three feature sets: A (mean income and raw CV), B (A plus all the STL features), C (mean income and residual CV only).

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/simulate_income.py
python src/decompose.py
python src/risk_proxy.py
python src/train_model.py
python src/plot_examples.py
```

Run from the project root. Outputs go to `data/` and `outputs/`.

## Results so far

Default rate is 28.8% overall: steady 13.6%, seasonal 38.0%, volatile 35.9%.

| Model | ROC-AUC |
|---|---|
| A. Mean income + raw CV | 0.710 |
| B. A + all STL features | 0.879 |
| C. Mean income + residual CV | 0.710 |

The 0.879 is misleading if read on its own. I ran ablations, and almost all of the jump from A to B comes from trend slope: mean income, raw CV and trend slope alone score about 0.876. The residual volatility, which is the feature my hypothesis is about, adds nothing over mean income here (A and C are identical).

Two other things I noticed:

- Seasonal workers default about as often as volatile ones. With a fixed repayment and a small cash buffer, a predictable monsoon dip still empties the buffer. Being predictable doesn't protect a borrower if the loan schedule ignores it.
- Part of the trend result is how I built the simulator. Each worker's trend is drawn once and keeps going into the loan year, so trend is easier to predict than it would be in real data.

So the original hypothesis is not supported by this first design. I'm treating that as a result and not something to tune away.

## Limitations

- The data is simulated. Real individual-level gig income with repayment outcomes is held by lenders and isn't public, so I generated income with patterns loosely based on published aggregates (JPMorgan Chase Institute on platform income volatility, Fairwork India, NITI Aayog). It reproduces plausible patterns, not real people.
- The default label rests on my assumptions (70% living costs, 25% payment, 3-week buffer). Results will move if these change, and I haven't tested how much yet.
- STL needs long histories. Most real gig workers wouldn't have two years on one platform.
- STL's smoothing uses data on both sides of each week, so it isn't causal. That's fine for analysis here, but a live scoring system would need a version that only uses past data, otherwise backtests would look better than real performance.
- It assumes a single 52-week cycle. Real income probably has weekly, monthly and annual patterns together.

## Next steps

1. Seasonally adjusted repayments. Build a second loan scenario where the repayment schedule follows the STL seasonal pattern (lower in the monsoon, higher at festivals). If predictable volatility really isn't risk, seasonal workers' default rate should fall close to steady workers' under this scheme, and residual volatility should become a more useful signal. This is the test that actually matches the hypothesis.
2. Sensitivity checks on the label: vary the living-cost share, payment share and buffer, and see which conclusions survive.
3. Fix the trend problem in the simulator (let trends change over time) and see whether trend slope still dominates.
4. Try a causal version of the decomposition, using only past weeks at each point, and compare it to full-window STL.
5. Test shorter histories (for example 26 to 52 weeks) with a simpler seasonal method, since that's closer to real gig tenure.
6. Try a gradient-boosted model next to logistic regression.
7. Write up the results as a short report, including the negative first result.

## Reading behind this

- Oyeyemi, O. O. Credit Access in the Gig Economy: Rethinking Creditworthiness in a Post-Employment Financial System. https://ssrn.com/abstract=5336043
- Anekadhana, N. (2025). The Labor Credit Scoring Model for the Gig Economy in Developing Countries. Research Square preprint, not peer reviewed. https://doi.org/10.21203/rs.3.rs-7198891/v1
- NITI Aayog (2022). India's Booming Gig and Platform Economy.
- Fairwork India Ratings 2024.
- JPMorgan Chase Institute, research on income volatility among platform workers.
- Cleveland, Cleveland, McRae and Terpenning (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. Practical guide: https://otexts.com/fpp3/stl.html

Gauri Mehrotra, B.Tech (AI), Newton School of Technology, Rishihood University
gaurimehrotra18@gmail.com
