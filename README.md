# Gig income volatility and credit risk

I'm testing one idea: that credit models treat all income volatility as risk, when for gig workers a lot of it is predictable (festival weeks, monsoon dips, weekly demand cycles). If that's true, splitting income into a predictable part and a leftover unexplained part should give a better risk signal than average income or raw variance.

This is an independent project, built alongside reading for my MBA applications. It uses simulated data, and I say why below.

## The question

Do features from an STL decomposition of weekly income, especially the residual volatility, predict loan default better than a baseline built on mean income and raw volatility? And if they don't, what does decomposing income actually help with?

## Pipeline

```
simulate_income.py   156 weeks of income for 500 simulated workers
        |
        |-- weeks 0-103   -->  decompose.py   -->  STL features per worker
        |                      (STL, period = 52)
        |
        '-- weeks 104-155 -->  loan simulation -->  default / no default
                               (risk_proxy.py, loan_sim.py)

features + outcome  -->  train_model.py, run_experiments.py,
                         sensitivity.py, lender_economics.py
```

The split in the middle is deliberate. Features only see the first two years of income. The default label only comes from the third year. That way the label is not computed from the same residual I'm using as a feature, which would make the comparison circular.

## What each script does

**simulate_income.py.** Builds weekly income as base level + slow trend + seasonal pattern (a festival bump around week 43, a monsoon dip around week 26) + noise + occasional income shocks. Workers come in three types:

| Type | Share | Seasonal swing | Shock chance per week |
|---|---|---|---|
| steady | 35% | small | 1.5% |
| seasonal | 35% | large | 1.5% |
| volatile | 30% | medium | 8% |

The `trend_std` argument controls how much each worker's income drifts up or down over time. Setting it to 0 gives a "no trend" world.

**decompose.py.** Runs robust STL on each worker's first 104 weeks and pulls out mean income, raw std and raw CV (the baseline), plus trend slope, seasonal amplitude, residual std, residual CV and the largest single residual. STL needs about two full seasonal cycles, which is why the observation window is 104 weeks.

**risk_proxy.py and loan_sim.py.** Simulate a loan taken at week 104. Weekly living costs are 70% of the worker's mean income, the weekly payment is 25%, and they start with a cash buffer of 3 payments. Each week cash changes by income minus living costs minus the payment. If cash goes below zero in the 52-week loan year, that worker defaults. `loan_sim.py` adds a second repayment schedule: instead of a fixed payment, the payment follows the worker's own STL seasonal pattern (lower in their weak season, higher in their strong one). Total repaid is the same under both.

**train_model.py.** First-pass comparison. Logistic regression on standardised features, ROC-AUC from repeated cross-validation.

**run_experiments.py.** The main test. Fixed vs seasonally adjusted repayment, over 5 random seeds, with and without trend. Four models: A (mean income + raw CV), B (A + all STL features), C (mean income + residual CV only), D (A + all STL features except trend).

**sensitivity.py.** Repeats the comparison over a grid of loan assumptions (living costs 65-75%, payment 20-30%, buffer 2-5 weeks).

**lender_economics.py.** Approves a worker if the model's predicted default probability is below the lender's break-even rate, margin / (margin + loss), then measures profit per applicant. The cutoff is set in advance, without looking at outcomes.

**policy_sim.py.** An earlier approval test. It picks its cutoff using the actual outcomes, so its approval rates are a best case. I keep it for reference only.

**plot_examples.py.** STL plots for one worker of each type.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# first pass
python src/simulate_income.py
python src/decompose.py
python src/risk_proxy.py
python src/train_model.py
python src/plot_examples.py

# later experiments (these build their own data, and take a few minutes each)
python src/run_experiments.py
python src/sensitivity.py
python src/lender_economics.py
```

Run from the project root. Outputs go to `data/` and `outputs/`.

## Results

### 1. First pass (train_model.py)

Default rate 28.8% overall: steady 13.6%, seasonal 38.0%, volatile 35.9%.

| Model | ROC-AUC |
|---|---|
| A. Mean income + raw CV | 0.710 |
| B. A + all STL features | 0.879 |
| C. Mean income + residual CV | 0.710 |

The jump to 0.879 is misleading. In the later experiments, B scores about 0.89 but D (the same features without trend slope) scores about 0.71. So the improvement comes from the trend, not from the volatility features my hypothesis is about.

### 2. Fixed vs seasonally adjusted repayment (run_experiments.py)

Default rate by worker type, averaged over 5 seeds:

| World | Schedule | Seasonal | Steady | Volatile |
|---|---|---|---|---|
| no trend | fixed | 23.0% | 0.7% | 21.8% |
| no trend | seasonal | 11.0% | 0.6% | 18.0% |
| with trend | fixed | 38.3% | 19.9% | 34.2% |
| with trend | seasonal | 28.2% | 19.9% | 32.0% |

ROC-AUC, averaged over 5 seeds (spread across seeds is about 0.01 to 0.035):

| World | Schedule | A | B | C | D |
|---|---|---|---|---|---|
| no trend | fixed | 0.694 | 0.697 | 0.667 | 0.699 |
| no trend | seasonal | 0.702 | 0.694 | 0.648 | 0.691 |
| with trend | fixed | 0.705 | 0.894 | 0.706 | 0.713 |
| with trend | seasonal | 0.704 | 0.896 | 0.706 | 0.709 |

### 3. Do the conclusions survive different assumptions? (sensitivity.py)

Single seed, 44 of 54 settings kept (the rest had default rates below 3% or above 97%).

- The seasonal schedule lowered seasonal workers' default rate in every kept setting. The average cut was 43% with no trend (range 2% to 73%) and 21% with trend (range 5% to 40%).
- STL features without trend beat the baseline AUC by more than 0.02 in only 41% (fixed) and 29% (seasonal) of no-trend settings, and 15% and 11% of with-trend settings. The average gain was between 0.00 and 0.014.

### 4. Lender economics (lender_economics.py)

3 seeds, 500 workers each. Profit is per applicant, in fractions of loan principal.

No trend, margin 10%, loss 60%:

| | Fixed | Seasonal |
|---|---|---|
| Baseline model A: share approved | 56% | 84% |
| Baseline model A: profit | 0.024 | 0.041 |
| Model D (STL features): profit | 0.023 | 0.041 |
| Approve everyone: profit | -0.007 | 0.034 |

With a 15% margin the baseline goes from 0.047 to 0.080, and with a 90% loss from 0.016 to 0.026. With trend the profits are close to zero (0.002 to 0.007 for the baseline at 10%/60%), so the effect is much weaker there.

### What I take from this

My original hypothesis, that residual volatility is a better risk signal than raw volatility, is not supported. Residual volatility on its own predicts default worse than raw volatility (model C below A in the no-trend runs), and the extra STL features add nothing beyond the baseline. My guess is that seasonal dips still hurt borrowers because living costs are fixed, so raw volatility keeps information that residual volatility throws away. I haven't tested that.

What does show up consistently is that how the loan is structured matters more than which score is used. Matching repayments to a worker's seasonal pattern cut seasonal workers' defaults in every setting I tried, and made lending to them profitable in the simulation. Under a fixed schedule, any volatility-aware score beats approving everyone. Under the seasonal schedule the portfolio is already safe enough that scoring adds little. So decomposition looks more useful for designing the loan than for ranking risk.

## Limitations

- The data is simulated. Real individual-level gig income with repayment outcomes is held by lenders and isn't public, so I generated income with patterns loosely based on published aggregates (JPMorgan Chase Institute on platform income volatility, Fairwork India, NITI Aayog). It reproduces plausible patterns, not real people.
- Each simulated worker's seasonal pattern repeats exactly every year. That is the best case for a seasonal repayment schedule. Real seasonality drifts, so the benefit is probably overstated. This is the biggest open weakness.
- The loan assumptions (living costs, payment size, buffer) are mine. The sensitivity check shows the direction holds but the size varies a lot.
- The economics ignore the cost of running seasonal schedules, and assume the same interest is collected either way. The margin and loss values are assumptions, not estimates.
- The sensitivity run used one seed per setting, so small AUC differences there are noisy.
- Each worker's trend is drawn once and continues into the loan year, which makes trend easier to predict than it would be in real data.
- STL needs about two years of history, and its smoothing is not causal (a week's estimate uses later weeks too). Here that's fine because the loan starts after the observation window, but a live system scoring at many dates would need a causal version.
- STL assumes a single 52-week cycle. Real income probably has weekly, monthly and annual patterns together.

## Next steps

1. Let each worker's seasonal amplitude and timing drift from year to year, and check whether the seasonal schedule still helps. This is the most important test left.
2. Let part of living costs flex with income, to test my guess about why seasonal swings still cause defaults after the schedule is adjusted.
3. Rerun the sensitivity grid over several seeds and add confidence intervals.
4. Rolling-origin backtest with a causal version of STL, using only past weeks.
5. Test short histories (26 to 52 weeks) with a simpler seasonal method, since that's closer to real gig tenure.
6. Write up the results as a short report, including that the original hypothesis failed.

## Reading behind this

- Oyeyemi, O. O. Credit Access in the Gig Economy: Rethinking Creditworthiness in a Post-Employment Financial System. https://ssrn.com/abstract=5336043
- Anekadhana, N. (2025). The Labor Credit Scoring Model for the Gig Economy in Developing Countries. Research Square preprint, not peer reviewed. https://doi.org/10.21203/rs.3.rs-7198891/v1
- NITI Aayog (2022). India's Booming Gig and Platform Economy.
- Fairwork India Ratings 2024.
- JPMorgan Chase Institute, research on income volatility among platform workers.
- Cleveland, Cleveland, McRae and Terpenning (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. Practical guide: https://otexts.com/fpp3/stl.html

Gauri Mehrotra, B.Tech (AI), Newton School of Technology, Rishihood University
gaurimehrotra18@gmail.com
