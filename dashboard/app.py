"""Streamlit dashboard:  streamlit run dashboard/app.py   (or: make dashboard)

Replays any past day with the frozen models and compares the forecast with the actual load, the official TSO forecast
and the weekly naive baseline. Paths come from EDF_DB_PATH / EDF_MODEL_DIR / EDF_REPORTS_DIR (defaults: the repo).
The warehouse is opened read-only, so the dashboard can run next to the API (but not during `make data`).
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure   # NOT pyplot: Streamlit runs the script in a worker thread and the
#                                        macOS backend can only create figures in the main thread

from edf import db, report, serving, viewdata
from edf.config import DB_PATH, MODEL_DIR, ROOT
from edf.features import load_dataset, load_official

st.set_page_config(page_title="German day-ahead load forecast", layout="wide")

DB = Path(os.environ.get("EDF_DB_PATH", DB_PATH))
MODELS = Path(os.environ.get("EDF_MODEL_DIR", MODEL_DIR))
REPORTS = Path(os.environ.get("EDF_REPORTS_DIR", ROOT / "reports"))


@st.cache_resource
def get_predictor(model_dir: str, db_path: str):
    return serving.Predictor(model_dir, db_path)


@st.cache_data(ttl=300)
def get_data(db_path: str):
    con = db.connect(db_path, read_only=True)
    try:
        return load_dataset(con), load_official(con)
    finally:
        con.close()


st.title("German day-ahead electricity demand forecast")
st.caption("Open data (SMARD / Bundesnetzagentur, Open-Meteo) - gradient boosting + neural network blend - "
           "90 % conformalised intervals")

tab_forecast, tab_results, tab_about = st.tabs(["Forecast explorer", "Results", "About and limitations"])

# ------------------------------------------------------------------------------------------------ explorer
with tab_forecast:
    try:
        predictor = get_predictor(str(MODELS), str(DB))
        df, official = get_data(str(DB))
        first, last = viewdata.selectable_days(df, predictor.meta["origin_hour"])
    except FileNotFoundError:
        st.error(f"No trained models in `{MODELS}`. Run `make train` first.")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a traceback
        st.error(f"Cannot load the data: {exc}")
        st.stop()

    c1, c2 = st.columns([1, 3])
    day = c1.date_input("Day to replay", value=last.date(), min_value=first.date(), max_value=last.date())
    show_naive = c1.checkbox("Show weekly naive", value=False)
    c1.caption("Replay with OBSERVED weather: the forecast uses the load known at 12:00 on the previous day. "
               "A live system would only have weather forecasts, so this is somewhat optimistic.")

    try:
        view = viewdata.day_view(predictor.predict_day(pd.Timestamp(day)), df["load"], official)
    except serving.ForecastNotAvailable as exc:
        c2.warning(str(exc))
        st.stop()
    m = viewdata.day_metrics(view)

    k1 = k2 = k3 = k4 = c1
    k1.metric("MAE blend (MWh)", f"{m['blend']:,.0f}")
    k2.metric("MAE official TSO (MWh)", f"{m['official']:,.0f}" if pd.notna(m["official"]) else "n/a")
    k3.metric("MAE weekly naive (MWh)", f"{m['naive']:,.0f}" if pd.notna(m["naive"]) else "n/a")
    k4.metric("Actual inside 90 % interval", f"{m['coverage'] * 100:.0f} %" if pd.notna(m["coverage"]) else "n/a")

    fig = Figure(figsize=(10, 4.8))
    ax = fig.subplots()
    x = view["ts_local"]
    ax.fill_between(x, view["lower_mwh"], view["upper_mwh"], alpha=0.25, label="90 % interval")
    ax.plot(x, view["actual_mwh"], color="black", lw=1.6, label="actual")
    ax.plot(x, view["point_mwh"], color="tab:red", lw=1.4, label="blend forecast")
    if view["official_mwh"].notna().any():
        ax.plot(x, view["official_mwh"], color="tab:green", lw=1.2, ls="--", label="official TSO")
    if show_naive:
        ax.plot(x, view["naive_mwh"], color="tab:blue", lw=1, ls=":", label="weekly naive")
    ax.set(ylabel="load (MWh per hour)", title=f"{pd.Timestamp(day):%A %d %B %Y} (local time)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    c2.pyplot(fig)
    with st.expander("Hourly values"):
        table = view[["ts_local", "actual_mwh", "point_mwh", "lower_mwh", "upper_mwh", "official_mwh",
                      "lgbm_mwh", "nn_mwh"]].copy()
        num = table.columns.drop("ts_local")
        table[num] = table[num].round(0)
        st.dataframe(table, use_container_width=True)

# ------------------------------------------------------------------------------------------------ results
with tab_results:
    st.markdown(report.build_results_md(REPORTS))
    figs = [p for p in ("04_interval_examples.png", "04_interval_coverage.png", "04_mae_by_hour.png",
                        "03_feature_importance.png") if (REPORTS / "figures" / p).exists()]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, name in zip(cols, figs[i:i + 2]):
            col.image(str(REPORTS / "figures" / name), caption=name)

# ------------------------------------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Task.** Forecast the hourly German grid load for day D using only what is known at 12:00 on D-1.

**Method.** Calendar (state-weighted public holidays, bridge days), solar position, weather and recent-load features;
LightGBM and a neural network blended 50/50; quantile regression with rolling conformal calibration for intervals.
Evaluation is a rolling-origin backtest with a locked final test set, compared with the official TSO forecast.

**Known limitations**
- The models are evaluated with **observed** weather of the target day; the official forecast was made with weather
  forecasts. The live mode of the API uses a weather forecast but was not validated.
- The information cutoff (12:00 on D-1) may differ from the one of the TSOs.
- Weeks after public holidays are harder: the lags of the previous weeks come from atypical days (e.g. January).
- Interval coverage is uneven across hours and months (under-coverage in the afternoon and in some winter months).
- One country, one target (total load), no extreme-event handling.

Data: Bundesnetzagentur | SMARD.de; weather by Open-Meteo.com (CC BY 4.0).
""")