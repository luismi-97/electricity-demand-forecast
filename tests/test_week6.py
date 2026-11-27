from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edf import report, viewdata
from tests.conftest import make_df

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


# ---- dashboard data logic ---------------------------------------------------------------------------------------
def test_selectable_days_needs_history_and_weather():
    df = make_df("2022-01-01", "2024-12-31 23:00")
    first, last = viewdata.selectable_days(df)
    assert first == pd.Timestamp("2022-01-01") + pd.Timedelta(days=400)
    assert last == pd.Timestamp("2024-12-31")                        # last local day fully covered by load AND weather
    with pytest.raises(ValueError):
        viewdata.selectable_days(make_df("2024-01-01", "2024-06-30 23:00"))
    with pytest.raises(ValueError):
        viewdata.selectable_days(make_df("2022-01-01", "2024-12-31 23:00", with_temp=False))


def test_day_view_and_metrics():
    idx = pd.date_range("2024-06-10", periods=24, freq="h", tz="UTC")
    load = pd.Series(50000.0, index=pd.date_range("2024-06-01", "2024-06-30 23:00", freq="h", tz="UTC"))
    fc = pd.DataFrame({"ts_utc": idx, "ts_local": idx.tz_convert("Europe/Berlin"), "point_mwh": 50100.0,
                       "lower_mwh": 49000.0, "upper_mwh": 51000.0, "lgbm_mwh": 50100.0, "nn_mwh": np.nan})
    off = pd.Series(50500.0, index=load.index)
    v = viewdata.day_view(fc, load, off)
    m = viewdata.day_metrics(v)
    assert m["blend"] == pytest.approx(100) and m["official"] == pytest.approx(500) and m["naive"] == pytest.approx(0)
    assert m["coverage"] == 1.0
    assert np.isnan(viewdata.day_metrics(viewdata.day_view(fc, load, None))["official"])


# ---- report generator -------------------------------------------------------------------------------------------
def _write_reports(d, final=True):
    summ = pd.DataFrame({"n": [17544] * 3, "MAE": [1217.0, 1953.0, 2498.0], "RMSE": [1613.0, 2472.0, 3939.0],
                         "MAPE_%": [2.34, 3.76, 4.89], "bias": [210.0, -115.0, -15.0], "MASE": [0.5, 0.8, 1.0]},
                        index=["blend_lgbm_nn", "official_tso", "naive_168h"])
    paired = pd.DataFrame({"mae_a": [1217.0], "mae_b": [1953.0], "diff_a_minus_b": [-736.0], "ci95_low": [-825.0],
                           "ci95_high": [-650.0], "share_days_a_better": [0.744], "n_days": [731]},
                          index=["blend_lgbm_nn vs official_tso"])
    iv = pd.DataFrame({"n": [16800, 16800], "nominal": [0.9, 0.9], "coverage": [0.659, 0.894],
                       "mean_width": [3000.0, 5813.0], "width_%_of_load": [6.9, 11.3]},
                      index=["raw quantile", "conformalised"])
    prefix = ("final_test", "final_test_paired", "final_test_intervals") if final else \
             ("week4_validation", "week4_paired_tests", "week4_intervals")
    for name, frame in zip(prefix, (summ, paired, iv)):
        frame.to_csv(d / f"{name}.csv")


def test_report_says_final_test_pending_when_missing(tmp_path):
    _write_reports(tmp_path, final=False)
    md = report.build_results_md(tmp_path)
    assert "Not evaluated yet" in md and "Validation 2023-2024" in md
    assert "1,217" in md and "37.7 % lower" in md                    # 1 - 1217/1953 = 37.7 %


def test_report_uses_final_test_numbers_when_available(tmp_path):
    _write_reports(tmp_path, final=False)
    _write_reports(tmp_path, final=True)
    md = report.build_results_md(tmp_path)
    assert "Final test (locked period" in md and "Not evaluated yet" not in md
    assert "-736" in md and "74 % of days" in md and "0.894" in md
    assert "observed" in md.lower()                                  # caveat always present


def test_readme_block_is_replaced_idempotently(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\nold\n{report.END}\noutro\n", encoding="utf-8")
    assert report.update_readme(readme, "NEW BLOCK")
    first = readme.read_text(encoding="utf-8")
    assert "NEW BLOCK" in first and "old" not in first and first.startswith("intro") and first.rstrip().endswith("outro")
    assert report.update_readme(readme, "NEW BLOCK") and readme.read_text(encoding="utf-8") == first
    plain = tmp_path / "x.md"
    plain.write_text("no markers", encoding="utf-8")
    assert report.update_readme(plain, "x") is False


# ---- dashboard app ----------------------------------------------------------------------------------------------
def test_dashboard_renders_with_models(warehouse, artifacts, monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("EDF_DB_PATH", str(warehouse[0]))
    monkeypatch.setenv("EDF_MODEL_DIR", str(artifacts[0]))
    monkeypatch.setenv("EDF_REPORTS_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "MAE blend (MWh)" in labels and "Actual inside 90 % interval" in labels
    assert any("Not evaluated yet" in md.value for md in at.markdown)


def test_dashboard_never_uses_pyplot(warehouse, artifacts, monkeypatch, tmp_path):
    """pyplot's global state breaks in Streamlit's worker thread on macOS ('Cannot create a GUI FigureManager
    outside the main thread'). The app must build figures with matplotlib.figure.Figure instead."""
    import matplotlib.pyplot as plt
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def boom(*a, **k):
        raise RuntimeError("pyplot must not be used in the dashboard")

    for name in ("subplots", "figure", "gcf", "gca", "plot", "fill_between"):
        monkeypatch.setattr(plt, name, boom)
    monkeypatch.setenv("EDF_DB_PATH", str(warehouse[0]))
    monkeypatch.setenv("EDF_MODEL_DIR", str(artifacts[0]))
    monkeypatch.setenv("EDF_REPORTS_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert "import matplotlib.pyplot" not in APP.read_text()


def test_dashboard_without_models_shows_helpful_error(warehouse, monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    empty = tmp_path / "no_models"
    empty.mkdir()
    monkeypatch.setenv("EDF_DB_PATH", str(warehouse[0]))
    monkeypatch.setenv("EDF_MODEL_DIR", str(empty))
    monkeypatch.setenv("EDF_REPORTS_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("make train" in e.value for e in at.error)