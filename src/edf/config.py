"""Project-wide configuration."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("EDF_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
DB_PATH = Path(os.environ.get("EDF_DB_PATH", DATA_DIR / "processed" / "warehouse.duckdb"))
MODEL_DIR = Path(os.environ.get("EDF_MODEL_DIR", ROOT / "models" / "v1"))
FIG_DIR = ROOT / "reports" / "figures"

# SMARD (Bundesnetzagentur) -----------------------------------------------
SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
SMARD_REGION = "DE"
# SMARD filter ids. 410 = total grid load (Netzlast), 4359 = residual load.
SMARD_SERIES = {
    "load": 410,
    "residual_load": 4359,
}
DEFAULT_START = "2015-01-01"

# Weather (Open-Meteo, ERA5 reanalysis) -----------------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"   # live mode of the API (not tested against the live service)
WEATHER_VARIABLES = ["temperature_2m", "shortwave_radiation", "wind_speed_10m"]
# Reanalysis lags real time by a few days; stay safely behind it.
WEATHER_LAG_DAYS = 7

# City -> (latitude, longitude, weight). Weights are APPROXIMATE city populations in
# millions and are only used to build a population-weighted temperature later.
# TODO(week 3): refine the choice of locations/weights and test sensitivity.
CITIES = {
    "berlin": (52.52, 13.41, 3.7),
    "hamburg": (53.55, 9.99, 1.9),
    "muenchen": (48.14, 11.58, 1.5),
    "koeln": (50.94, 6.96, 1.1),
    "frankfurt": (50.11, 8.68, 0.77),
    "stuttgart": (48.78, 9.18, 0.63),
}

# Evaluation protocol --------------------------------------------------------
TZ = "Europe/Berlin"
ORIGIN_HOUR = 12              # day-ahead gate closure: forecasts for day D are issued at 12:00 local on D-1
VALIDATION_PERIOD = ("2023-01-01", "2024-12-31")   # model development / selection
FINAL_TEST_START = "2025-01-01"                    # locked until the very end of the project
