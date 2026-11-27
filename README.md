# Day-ahead electricity demand forecasting for Germany
 
End-to-end forecasting project on open German energy data: ingestion, SQL storage, a rigorous time-series evaluation
protocol with a locked test set, gradient boosting and neural-network models, calibrated prediction intervals, an API
and a dashboard.
 
**Task:** forecast the 24 hourly values of the German grid load for day D using only what is known at 12:00 on D-1
(day-ahead gate closure), and compare the result with simple baselines and with the official TSO day-ahead forecast.
 
## Highlights
 
- **No leakage by construction:** one function builds the features for training and for live forecasts, and unit tests
  check that corrupting every value after the forecast cutoff leaves the features unchanged.
- **Honest evaluation:** rolling-origin backtest (one forecast per day, models refitted every 30 days), a final test
  set that the code refuses to touch until the configuration is frozen, and paired bootstrap tests that resample
  whole days.
- **Calibrated uncertainty:** quantile regression with rolling conformal calibration that only uses outcomes known at
  the forecast origin.
- **Production-minded:** frozen configuration with a hash, saved artifacts, FastAPI service, Docker image, Makefile,
  CI, a Streamlit dashboard and 60+ unit tests.
<!-- RESULTS:START -->
## Results

### Final test (locked period, evaluated once)

- MAE of **Blend (LightGBM + NN)**: 1,299 MWh vs. 1,980 MWh for the official TSO forecast (34.4 % lower error); paired bootstrap over days: difference -681 MWh (95 % CI -759 to -603), lower error on 75 % of days
- MAE of **Blend (LightGBM + NN)**: 1,299 MWh vs. 2,697 MWh for the weekly naive baseline (51.9 % lower error); paired bootstrap over days: difference -1,399 MWh (95 % CI -1,570 to -1,230), lower error on 84 % of days

Point forecasts (MAE, RMSE and bias in MWh; MAPE in %; MASE scaled by the weekly naive):

| Model | n | MAE | RMSE | MAPE_% | bias | MASE |
|---|---|---|---|---|---|---|
| Blend (LightGBM + NN) | 16,224 | 1,299 | 1,682 | 2.45 | -278 | 0.54 |
| Neural network | 16,224 | 1,346 | 1,769 | 2.54 | -300 | 0.56 |
| LightGBM | 16,224 | 1,372 | 1,783 | 2.59 | -256 | 0.57 |
| LightGBM (quantile, q50) | 16,224 | 1,403 | 1,827 | 2.65 | -263 | 0.58 |
| Official TSO forecast | 16,224 | 1,980 | 2,547 | 3.79 | +316 | 0.82 |
| Naive (same hour last week) | 16,224 | 2,697 | 3,986 | 5.13 | -95 | 1.11 |

90 % prediction intervals (nominal coverage 0.90):

| Interval | n | nominal | coverage | mean_width | width_%_of_load |
|---|---|---|---|---|---|
| raw quantile | 16,224 | 0.90 | 0.672 | 3,818 | 7.3 |
| conformalised | 15,480 | 0.90 | 0.901 | 6,421 | 12.3 |

### Validation 2023-2024 (used for model development)

- MAE of **Blend (LightGBM + NN)**: 1,217 MWh vs. 1,953 MWh for the official TSO forecast (37.7 % lower error); paired bootstrap over days: difference -736 MWh (95 % CI -825 to -650), lower error on 74 % of days
- MAE of **Blend (LightGBM + NN)**: 1,217 MWh vs. 2,498 MWh for the weekly naive baseline (51.3 % lower error); paired bootstrap over days: difference -1,281 MWh (95 % CI -1,460 to -1,110), lower error on 79 % of days

Point forecasts (MAE, RMSE and bias in MWh; MAPE in %; MASE scaled by the weekly naive):

| Model | n | MAE | RMSE | MAPE_% | bias | MASE |
|---|---|---|---|---|---|---|
| Blend (LightGBM + NN) | 17,544 | 1,217 | 1,613 | 2.34 | +210 | 0.50 |
| LightGBM | 17,544 | 1,264 | 1,686 | 2.43 | +161 | 0.52 |
| LightGBM (quantile, q50) | 17,544 | 1,295 | 1,737 | 2.50 | +126 | 0.54 |
| Neural network | 17,544 | 1,314 | 1,752 | 2.52 | +258 | 0.54 |
| Official TSO forecast | 17,544 | 1,953 | 2,472 | 3.76 | -115 | 0.81 |
| Naive (same hour last week) | 17,544 | 2,498 | 3,939 | 4.89 | -15 | 1.03 |

90 % prediction intervals (nominal coverage 0.90):

| Interval | n | nominal | coverage | mean_width | width_%_of_load |
|---|---|---|---|---|---|
| raw quantile | 17,544 | 0.90 | 0.659 | 3,599 | 6.9 |
| conformalised | 16,800 | 0.90 | 0.894 | 5,813 | 11.3 |

> Caveats: the models see the **observed** weather of the target day (the official forecast was made with weather forecasts) and use load up to 12:00 on D-1, which may differ from the TSOs' information cutoff. Read the comparison as 'under this protocol', not as a like-for-like contest.

<!-- RESULTS:END -->
 
## Method
 
**Data**
 
| Source | Content | Notes |
|---|---|---|
| [SMARD](https://www.smard.de) (Bundesnetzagentur) | Grid load (hourly), official day-ahead forecast (CSV export) | Attribution: *Bundesnetzagentur \| SMARD.de*; check the site's current terms |
| [Open-Meteo](https://open-meteo.com) | Hourly ERA5 reanalysis: temperature, solar radiation, wind | CC BY 4.0, non-commercial use. *Weather data by Open-Meteo.com* |
 
**Features** (`make_features`, identical for training and prediction)
- *Calendar:* hour, weekday, month, day of year; population-weighted share of states with a public holiday, day
  before/after a holiday, bridge days, Christmas-New Year period.
- *Solar:* approximate solar elevation at the centre of Germany.
- *Weather (target hours, observed):* population-weighted temperature, radiation, wind, daily and 72-hour means,
  heating/cooling degrees.
- *Recent load (strictly before the cutoff):* same hour 168 h and 336 h earlier, same hour on the latest available
  day, 24 h and 168 h rolling levels ending at the cutoff, forecast horizon.
**Models** (`src/edf/models/`): weekly seasonal naive; ridge regression on hour-of-week, temperature and season;
LightGBM; MLP in TensorFlow predicting the change versus last week; a 50/50 blend of LightGBM and the network;
LightGBM quantile regression (5/50/95 %) for intervals.
 
**Evaluation protocol**
- Forecast for day D issued at 12:00 local time on D-1 (horizon about 13-36 h), DST days handled (23/25 hours).
- Model development on the validation period 2023-2024; everything from 2025-01-01 is locked in `backtest()` until the
  one-off final test (`make final-test`, which needs `--confirm-final-test`).
- Metrics: MAE, RMSE, MAPE, bias, MASE; differences tested with a paired bootstrap over days; intervals judged by
  coverage, width and pinball loss.
- The configuration used in the final test lives in `src/edf/frozen.py` (hash printed with the results); unit tests
  check that it matches the setup used in validation.
## Limitations
 
- **Oracle weather:** features use the *observed* weather of the target day, whereas the official forecast was made
  with weather forecasts. A sensitivity analysis (temperature noise up to 3 C) changed the validation MAE by only
  about 2 %, but the noise levels are assumptions, not measured forecast errors. The API's live mode uses a real
  weather forecast and was not validated.
- **Information cutoff:** the models use load up to 12:00 on D-1; the TSOs' cutoff may be earlier or later.
- **Holiday-contaminated lags:** weeks after public holidays are harder because the weekly lags come from atypical
  days (visible in January 2024). Not fixed in v1.
- **Uneven interval coverage:** after conformal calibration the overall coverage is close to nominal, but the
  intervals under-cover in the afternoon and in some winter months and over-cover at night.
- **Scope:** one country, total load only, no extreme-event handling; national public holidays are state-weighted but
  school holidays are not modelled. Deployed artifacts are trained on all data, including the former test period.
## Quickstart
 
```bash
python -m venv .venv && source .venv/bin/activate
make install                                   # pip install -e ".[dev,nn,api,dashboard]"
make data                                      # load + weather (SMARD, Open-Meteo)
make official CSV=data/raw/smard_forecast.csv  # official forecast: SMARD download centre -> CSV
make test
make validate                                  # notebooks 01-04: EDA, baselines, GBM, NN + intervals
git commit -am "freeze v1" && git tag v1-frozen
make final-test                                # ONE-SHOT evaluation of the frozen configuration
make report                                    # refresh the results block of this README from the CSVs
make train                                     # fit on all data -> models/v1
make serve                                     # API on http://localhost:8000  (docs at /docs)
make dashboard                                 # Streamlit on http://localhost:8501
```
 
Stop the API and the dashboard before `make data` / `make official`: DuckDB allows one writer at a time.
 
## API and dashboard
 
| Endpoint | What it returns |
|---|---|
| `GET /health` | service and model status |
| `GET /model` | version, configuration hash, training end, interval settings |
| `GET /forecast?date=YYYY-MM-DD[&weather=archive\|forecast]` | hourly point forecast, 90 % interval and model components |
 
Days whose inputs are not yet available return `409` with the missing timestamp instead of a guess. The Docker image
(`make docker-build`, `make docker-run`; add `WITH_NN=1` for TensorFlow) serves the API with `data/` and `models/`
mounted. The dashboard replays any past day against the actual load, the official forecast and the naive baseline,
and shows the results tables. Model files are joblib pickles: only load artifacts you trained yourself.
 
## Repository layout
 
```
src/edf/        config, SMARD/weather/official-forecast clients, DuckDB layer, features, evaluation,
                frozen.py (v1 configuration), train_final.py, serving.py, api.py, report.py, viewdata.py
src/edf/models/ interface, baselines, LightGBM (point + quantile), neural network
notebooks/      01_eda ... 05_final_test (percent-format scripts: run as notebooks or as scripts)
dashboard/      Streamlit app
tests/          unit tests (network calls mocked)
Dockerfile, Makefile, .github/workflows/ci.yml, docs/blog_post.md
```
 
## Future work
 
Holiday-aware lags (flag or replace lags that fall on holidays), hour-specific conformal calibration, validation with
archived weather *forecasts* instead of reanalysis, school holidays and other exogenous drivers, multi-day horizons.
 
## Licence
 
Code: MIT (see `LICENSE`). Data: see the sources above.