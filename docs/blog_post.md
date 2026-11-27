# Forecasting Germany's electricity demand a day ahead (draft)

## Why it matters

Every hour, grid operators and energy traders need to know how much electricity Germany will use tomorrow. A forecast
that is a few percent off means expensive balancing power or a mispriced position. The operators publish an official
day-ahead forecast; I wanted to know how far open data and a carefully evaluated machine-learning pipeline could get.

## What I built

A pipeline on open data (grid load from the Bundesnetzagentur's SMARD portal, weather from Open-Meteo) that predicts
the 24 hourly values of tomorrow's load using only what is known at noon today, the same moment as the day-ahead
market closes. It has an API, a dashboard and a Docker image, and the code is public: [link to repository].

- **Features** that encode how Germany actually behaves: state-weighted public holidays, bridge days, daylight,
  weather, and recent load.
- **Models:** gradient boosting and a small neural network, blended 50/50, plus quantile regression for 90 % intervals.
- **Evaluation I would trust:** one forecast per day, rolling retraining, a final test period locked away until the
  configuration was frozen, and a comparison with the official forecast tested with a bootstrap over days.

## What I found

- On the validation period (2023-2024) the blend's error was 1,217 MWh on average versus 1,953 MWh for the official
  forecast and 2,498 MWh for "same hour last week". [FINAL: the same comparison on the locked test period.]
- The biggest single lesson was about **drift**: models without recent-load features were biased by about 2.5 GW
  because German consumption has fallen since the training years. Adding the load of the previous weeks fixed it.
- Calibrating intervals matters: raw quantile intervals labelled "90 %" covered only 66 % of hours; with conformal
  calibration they covered about 89 %.

## What this does not show

- My models use the **observed** weather of the forecast day; the official forecast was made with weather forecasts.
  A sensitivity test suggests this is not the main driver, but the noise levels I tested were assumptions.
- The information cutoff may differ from the operators', so this is not a like-for-like contest.
- Weeks after public holidays are still hard, and the intervals under-cover in the afternoon. Both are documented in
  the repository as known limitations and next steps.

## What I would do next

Holiday-aware lags, hour-specific interval calibration, and evaluation with archived weather forecasts.

---

## Short LinkedIn version

I built an open-data pipeline that forecasts Germany's electricity demand a day ahead, using only what is known at noon
the day before. The part I am proudest of is the evaluation: rolling retraining, a test period locked until the model
was frozen, and a bootstrap comparison against the official operators' forecast. On the 2023-24 validation period the
error was about [FINAL: x] % lower than the official forecast under my protocol [FINAL: update with the locked test
result]. I also document what it does not prove (observed weather, information cutoff). Code, API and dashboard: [link].
#datascience #machinelearning #energy #timeseries
