# Gig income volatility and credit risk

I started with one idea: that credit models treat all income volatility as risk, when for gig workers a lot of it is predictable (festival weeks, monsoon dips, weekly demand cycles). If that's true, splitting income into a predictable part and a leftover unexplained part should give a better risk signal than average income or raw variance.

I tested that, and it was wrong. This repo is the record of how I tested it, what held up, and what I found instead. It's an independent project, built alongside reading for my MBA applications, and it uses simulated data (the reason is below).

## The question

Do features from an STL decomposition of weekly income, especially the residual volatility, predict loan default better than a baseline built on mean income and raw volatility? If they don't, what does predict default, and what is decomposition actually good for?

## Pipeline

```
simulate_income.py   156 weeks of income for simulated gig workers
        |
        |-- weeks 0-103   -->  features (what the lender sees)
        |                        decompose.py    STL-only features        (stage 1)
        |                        features_ml.py   baseline + cash-flow +
        |                                         simple trend + STL      (stage 2)
        |
        '-- weeks 104-155 -->  loan simulation (risk_proxy.py, loan_sim.py)
                                 fixed or seasonally adjusted repayments,
                                 normal year or a 12-week downturn
                                 --> default / no default

Stage 1:  train_model.py, run_experiments.py, sensitivity.py, lender_economics.py
Stage 2:  build_ml_dataset.py  -->  ml_models.py, feature_ablation.py, finance_analysis.py
```

Features only see the first two years of income. The default label only comes from the third year. That way the label is never computed from the same residual I'm using as a feature, which would make the comparison circular.

## The simulation

Weekly income is base level + slow trend + seasonal pattern (a festival bump around week 43, a monsoon dip around week 26) + noise + occasional income shocks. Workers come in three types:

| Type | Share | Seasonal swing | Shock chance per week |
|---|---|---|---|
| steady | 35% | small | 1.5% |
| seasonal | 35% | large | 1.5% |
| volatile | 30% | medium | 8% |

`trend_std` controls how much each worker's income drifts over time. Setting it to 0 gives a "no trend" world, and I report both.

The loan: weekly living costs are 70% of the worker's mean income, the payment is 25%, and they start with a buffer of 3 payments. Each week cash changes by income minus living costs minus the payment. If cash goes below zero in the 52-week loan year, the worker defaults. A second schedule makes the payment follow the worker's own STL seasonal pattern (lower in their weak season, higher in their strong one), with the same total repaid.

## What each script does

**Stage 1 (STL-only features)**

- `simulate_income.py` generates the workers.
- `decompose.py` runs robust STL on weeks 0-103 and extracts mean income, raw CV, trend slope, seasonal amplitude, residual CV and the largest residual.
- `risk_proxy.py` and `loan_sim.py` simulate the loan and build the default label, for both repayment schedules.
- `train_model.py` is the first-pass model comparison.
- `run_experiments.py` compares fixed against seasonally adjusted repayment over 5 seeds, with and without trend.
- `sensitivity.py` repeats it over a grid of loan assumptions (living costs 65-75%, payment 20-30%, buffer 2-5 weeks).
- `lender_economics.py` approves workers whose predicted default probability is below the break-even rate, margin / (margin + loss), and measures profit per applicant.
- `policy_sim.py` is an early approval test that picks its cutoff using the actual outcomes, so its numbers are a best case. I keep it for reference only.
- `plot_examples.py` draws STL plots for one worker of each type.

**Stage 2 (ML and finance)**

- `features_ml.py` builds four feature families: baseline (mean income, raw CV), cash-flow statistics a lender could compute without any decomposition (income floor, worst 4 and 12-week stretches, downside deviation, skew), a simple trend from a straight-line fit on raw income, and the STL features (trend, seasonal strength, residual CV, bad-week count, residual autocorrelation).
- `build_ml_dataset.py` simulates 3,000 workers per world, computes all features, and labels defaults under both schedules, for a normal year and for a stress year (a 10% income slump for 12 weeks, landing in the monsoon dip). It caches the result in `data/`.
- `ml_models.py` compares logistic regression (L2 and L1), random forest and gradient boosting against the baseline, plus an "oracle" that is also told each worker's true type, as an upper bound. It reports AUC, Gini, KS, Brier score and bootstrap confidence intervals, and draws ROC, calibration and feature-importance plots.
- `feature_ablation.py` asks which feature family carries the information, using repeated cross-validation.
- `finance_analysis.py` looks at profit curves, who gets wrongly rejected, and a downturn stress test.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# stage 1
python src/simulate_income.py
python src/decompose.py
python src/risk_proxy.py
python src/train_model.py
python src/plot_examples.py
python src/run_experiments.py
python src/sensitivity.py
python src/lender_economics.py

# stage 2
python src/build_ml_dataset.py            # about 5 minutes, caches data/ml_*.csv
python src/feature_ablation.py
python src/finance_analysis.py
python src/ml_models.py no_trend fixed    # run all four combinations of
python src/ml_models.py no_trend seasonal # {no_trend, with_trend} x {fixed, seasonal}
python src/ml_models.py with_trend fixed
python src/ml_models.py with_trend seasonal
```

Run from the project root. Outputs go to `data/` and `outputs/`.

## Results

### Stage 1: STL features only

With only the STL features, the model looked like a success and then wasn't. Default rate was 28.8% overall (steady 13.6%, seasonal 38.0%, volatile 35.9%). AUC went from 0.710 (mean income + raw CV) to 0.879 with all STL features. But the same features without trend slope score about 0.71, so the whole jump came from the trend, not from the volatility features my hypothesis is about.

Seasonally adjusted repayment (default rate by worker type, 5 seeds):

| World | Schedule | Seasonal | Steady | Volatile |
|---|---|---|---|---|
| no trend | fixed | 23.0% | 0.7% | 21.8% |
| no trend | seasonal | 11.0% | 0.6% | 18.0% |
| with trend | fixed | 38.3% | 19.9% | 34.2% |
| with trend | seasonal | 28.2% | 19.9% | 32.0% |

The sensitivity grid (single seed, 44 of 54 settings kept) found that the seasonal schedule lowered seasonal workers' default rate in every kept setting, by 43% on average with no trend (range 2% to 73%) and 21% with trend (5% to 40%). In the lender economics test, profit per applicant rose from 0.024 to 0.041 under the seasonal schedule (no trend, margin 10%, loss 60%), and approving everyone lost money under the fixed schedule.

### Stage 2: ML models on richer features (3,000 workers per world)

**Which feature family carries the information?** Cross-validated AUC, gradient boosting (logistic regression shows the same pattern, see `outputs/ablation_results.csv`):

| Features | no trend, fixed | no trend, seasonal | trend, fixed | trend, seasonal |
|---|---|---|---|---|
| Baseline (mean income, raw CV) | 0.745 | 0.702 | 0.710 | 0.700 |
| Mean income + residual CV only | 0.653 | 0.606 | 0.705 | 0.692 |
| + cash-flow statistics | 0.791 | 0.724 | 0.744 | 0.731 |
| + cash-flow + simple trend (no decomposition) | 0.790 | 0.722 | 0.913 | 0.918 |
| + cash-flow + STL | 0.789 | 0.729 | 0.906 | 0.907 |

**Do fancier models help?** Held-out test set (900 workers), no trend, fixed schedule:

| Model | AUC | 95% interval | Gini | KS |
|---|---|---|---|---|
| Baseline logistic regression | 0.707 | 0.658-0.753 | 0.414 | 0.324 |
| Logistic regression, all features | 0.806 | 0.767-0.840 | 0.612 | 0.491 |
| Logistic regression (L1) | 0.805 | 0.766-0.839 | 0.610 | 0.485 |
| Random forest | 0.815 | 0.779-0.847 | 0.630 | 0.526 |
| Gradient boosting | 0.803 | 0.766-0.837 | 0.605 | 0.469 |
| Oracle (told each worker's true type) | 0.809 | 0.772-0.843 | 0.618 | 0.469 |

With trend, every full-feature model lands between 0.905 and 0.912 against an oracle of 0.916. With no trend and the seasonal schedule there is a little more room: gradient boosting 0.720, oracle 0.748, baseline 0.656.

**Who gets wrongly rejected?** Share of workers who would have repaid that the model approves, using a cutoff set in advance at the break-even default rate (14.3%), no trend:

| Schedule | Model | steady | seasonal | volatile |
|---|---|---|---|---|
| fixed | baseline | 0.72 | 0.68 | 0.43 |
| fixed | gradient boosting | 0.98 | 0.37 | 0.50 |
| seasonal | baseline | 0.99 | 0.95 | 0.77 |
| seasonal | gradient boosting | 0.99 | 0.80 | 0.71 |

**Profit and stress.** Profit per applicant, as a fraction of loan principal, with a margin of 10% and a loss of 60% on defaults:

| World | Schedule | Approve everyone | Best cutoff (approval share) | Stress year: default among approved, predicted -> realised | Profit, normal -> stress |
|---|---|---|---|---|---|
| no trend | fixed | -0.002 | 0.037 (52%) | 4.8% -> 15.4% | 0.034 -> -0.005 |
| no trend | seasonal | 0.044 | 0.048 (83%) | 4.9% -> 15.6% | 0.048 -> -0.008 |
| trend | fixed | -0.109 | 0.033 (56%) | 5.0% -> 12.7% | 0.032 -> 0.006 |
| trend | seasonal | -0.079 | 0.040 (52%) | 4.4% -> 12.2% | 0.039 -> 0.009 |

The stress year replays the loan with a 10% income slump for 12 weeks in the monsoon dip, after the model has already made its approvals on a normal year.

## What I take from this

1. **My original hypothesis is rejected.** A model with only mean income and residual volatility is the weakest feature set almost everywhere (0.653 against 0.745 for the baseline, no trend, fixed). Taking out the seasonal part throws information away.
2. **The income floor is what predicts default.** The strongest feature by a wide margin is the 10th-percentile week relative to mean income. Shuffling it costs 0.18 AUC, against 0.05 for mean income. Default here is a cash buffer running out, so what matters is the left tail relative to the lender's obligations, not volatility in general. This is partly built in, because I defined default as a buffer breach.
3. **The decomposition is not what adds value.** With no trend, adding STL features to the cash-flow statistics changes nothing (0.789 against 0.791). With trend, STL features take AUC from about 0.74 to 0.91, but a straight-line slope on raw income does just as well (0.913 against 0.906). STL's gain is trend detection, not seasonal decomposition.
4. **More complex models don't pay.** Logistic regression, random forest and gradient boosting are indistinguishable on the main setting, and all sit at the oracle. The features already capture nearly everything that can be seen, and the remaining error is future randomness.
5. **A more accurate model can be less inclusive.** Under fixed repayments, the gradient boosting model approves 98% of would-repay steady workers but only 37% of would-repay seasonal workers, because it has correctly learned that seasonal workers are riskier on that product. Under the seasonal schedule 80% get approved. The lever for inclusion is how the loan is built, not which model scores it.
6. **The schedule fixes predictable risk, not systemic risk.** Matching repayments to a worker's own seasonal pattern raised profit and approvals (best profit 0.037 to 0.048, approvals 52% to 83%, no trend). In a 12-week downturn, defaults among approved workers were the same under both schedules (about 15.5%), because the schedule approves more workers and carries the same exposure at higher volume.
7. **Normal-year models understate downturn risk.** The model predicted about 5% default among the workers it approved, and the stress year produced 12% to 16%. That gap is model risk, and it doesn't depend on which features are used.
8. **Scoring matters most when income can drift.** With trend, approving everyone loses 0.109 per applicant and a model earns +0.033. With no trend under the seasonal schedule, scoring adds only 0.004 over approving everyone.

Decomposition, then, is not the feature that predicts default. If it's useful, it's for designing the loan.

## Limitations

- The data is simulated. Real individual-level gig income with repayment outcomes is held by lenders and isn't public, so I generated income with patterns loosely based on published aggregates (JPMorgan Chase Institute, Fairwork India, NITI Aayog). It reproduces plausible patterns, not real people.
- Default is defined as running out of cash, so features that describe the downside tail were always going to do well. Real defaults also depend on behaviour I haven't modelled.
- Each worker's seasonal pattern repeats exactly every year, which is the best case for a seasonal repayment schedule. Real seasonality drifts, so the benefit is probably overstated.
- The simulated trend is a straight line and the window is exactly two seasonal cycles, which is the best case for a straight-line slope. A bending trend or a window that isn't a whole number of cycles could favour STL. I haven't tested that.
- The loan assumptions (living costs, payment size, buffer) and the economics (10% margin, 60% loss, no cost of running seasonal schedules) are mine and not estimated from data.
- The stage 2 results use one dataset seed. The bootstrap intervals cover test-sample noise, not variation across simulations, and the profit, inclusion and stress tables have no confidence intervals. The stress scenario is a single hand-picked case.
- STL needs about two years of history, and its smoothing is not causal (a week's estimate uses later weeks too). That is fine here because the loan starts after the observation window, but a live system scoring at many dates would need a causal version.

## Next steps

1. Test STL against a straight-line slope when the window isn't a whole number of cycles (for example 90 weeks) and when the trend bends. This is where a decomposition should earn its place, if anywhere.
2. Let each worker's seasonal amplitude and timing drift from year to year, and check whether the seasonal schedule still helps.
3. Let part of living costs flex with income, to test whether fixed costs explain why seasonal swings still cause defaults after the schedule is adjusted.
4. Repeat the stage 2 analysis over several dataset seeds and add confidence intervals to the profit, inclusion and stress tables.
5. Rolling-origin backtest with a causal version of STL, using only past weeks.
6. Short histories (26 to 52 weeks) with simpler seasonal methods, which is closer to real gig tenure.
7. Write up the results as a short report, including that the original hypothesis failed.

## Reading behind this

- Oyeyemi, O. O. Credit Access in the Gig Economy: Rethinking Creditworthiness in a Post-Employment Financial System. https://ssrn.com/abstract=5336043
- Anekadhana, N. (2025). The Labor Credit Scoring Model for the Gig Economy in Developing Countries. Research Square preprint, not peer reviewed. https://doi.org/10.21203/rs.3.rs-7198891/v1
- NITI Aayog (2022). India's Booming Gig and Platform Economy.
- Fairwork India Ratings 2024.
- JPMorgan Chase Institute, research on income volatility among platform workers.
- Cleveland, Cleveland, McRae and Terpenning (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. Practical guide: https://otexts.com/fpp3/stl.html

Gauri Mehrotra, B.Tech (AI), Newton School of Technology, Rishihood University
gaurimehrotra18@gmail.com