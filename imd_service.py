"""
imd_service.py
─────────────────────────────────────────────────────────────────────────────
Fetches real-time / historical meteorological data aligned with IMD
(India Meteorological Department) observations for Dakshina Kannada.

Data Source: Open-Meteo Historical/Archive API
  - Backed by ERA5-Land (ECMWF reanalysis, 0.1° resolution)
  - ERA5 is the same global reanalysis model that IMD ingests for analysis
  - Free, no API key required, CC BY 4.0 license
  - Data available up to the current day (typically ~1-2 hour lag for today)
  - Docs: https://open-meteo.com/en/docs/historical-weather-api

Station reference: Dakshina Kannada (Mangaluru)
  Lat: 12.87°N, Lon: 74.88°E
"""

import requests
from datetime import datetime, date
from typing import Optional
import logging

logger = logging.getLogger(__name__)

LATITUDE  = 12.87
LONGITUDE = 74.88
TIMEZONE  = "Asia/Kolkata"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "precipitation_sum",        # → prectotcorr (mm/day)
    "surface_pressure_mean",    # → ps (kPa)
    "temperature_2m_mean",      # → t2m (°C)
    "temperature_2m_max",       # → t2m_max (°C)
    "temperature_2m_min",       # → t2m_min (°C)
    "relative_humidity_2m_mean",# → rh2m (%)
    "wind_speed_10m_mean",      # → ws2m (m/s after conversion)
    "wind_direction_10m_dominant",  # → wd2m (degrees)
    "shortwave_radiation_sum",  # → allsky_sfc_sw_dwn (MJ/m²/day)
]


def _to_date_str(dt) -> str:
    """Convert date/datetime → YYYY-MM-DD."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    return str(dt)


def fetch_daily_data(target_date: datetime) -> Optional[dict]:
    """
    Fetch a full day's meteorological data from Open-Meteo for Dakshina Kannada.

    Returns a dict with CSV-compatible column names, e.g.:
        {
            'prectotcorr': 12.5,   # mm/day
            'ps':          99.2,   # kPa
            't2m':         27.8,   # °C
            ...
        }
    Returns None if data is unavailable or an error occurs.
    """
    date_str = _to_date_str(target_date)

    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "start_date": date_str,
        "end_date":   date_str,
        "daily":      ",".join(DAILY_VARS),
        "timezone":   TIMEZONE,
    }

    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.warning("Open-Meteo API timed out for date %s", date_str)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("Open-Meteo API request failed: %s", exc)
        return None
    except ValueError:
        logger.warning("Open-Meteo API returned invalid JSON.")
        return None

    try:
        daily = data["daily"]
        # Find the index for our date
        times = daily.get("time", [])
        if date_str not in times:
            logger.info("Date %s not in Open-Meteo response times: %s", date_str, times)
            return None
        idx = times.index(date_str)

        def get_val(key):
            vals = daily.get(key, [])
            return vals[idx] if idx < len(vals) else None

        precip    = get_val("precipitation_sum")
        pressure  = get_val("surface_pressure_mean")    # hPa
        t2m       = get_val("temperature_2m_mean")
        t2m_max   = get_val("temperature_2m_max")
        t2m_min   = get_val("temperature_2m_min")
        rh2m      = get_val("relative_humidity_2m_mean")
        ws_kmh    = get_val("wind_speed_10m_mean")      # km/h → m/s
        wd2m      = get_val("wind_direction_10m_dominant")
        sw        = get_val("shortwave_radiation_sum")

        # Validate all fields are present
        fields = [precip, pressure, t2m, t2m_max, t2m_min, rh2m, ws_kmh, wd2m, sw]
        if any(v is None for v in fields):
            logger.info("One or more fields missing for %s in Open-Meteo response.", date_str)
            return None

        return {
            "prectotcorr":     round(float(precip),           2),
            "ps":              round(float(pressure) / 10.0,  4),   # hPa → kPa
            "t2m":             round(float(t2m),              2),
            "t2m_max":         round(float(t2m_max),          2),
            "t2m_min":         round(float(t2m_min),          2),
            "rh2m":            round(float(rh2m),             2),
            "ws2m":            round(float(ws_kmh) / 3.6,    4),   # km/h → m/s
            "wd2m":            round(float(wd2m),             1),
            "allsky_sfc_sw_dwn": round(float(sw),             4),
        }

    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Error parsing Open-Meteo response: %s", exc)
        return None


def fetch_rainfall_only(target_date: datetime) -> Optional[float]:
    """
    Lightweight variant — fetches only precipitation_sum (rainfall mm/day).
    Returns rainfall in mm, or None if unavailable.
    """
    date_str = _to_date_str(target_date)
    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": date_str,
        "end_date":   date_str,
        "daily":      "precipitation_sum",
        "timezone":   TIMEZONE,
    }

    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        times = data["daily"]["time"]
        vals  = data["daily"]["precipitation_sum"]
        if date_str in times:
            idx = times.index(date_str)
            val = vals[idx]
            if val is not None:
                return round(float(val), 2)
        return None
    except Exception as exc:
        logger.warning("Open-Meteo rainfall fetch failed: %s", exc)
        return None


def get_data_availability_info() -> dict:
    """
    Returns metadata about Open-Meteo data availability.
    Open-Meteo ERA5 typically covers up to the current day.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    val = fetch_rainfall_only(datetime.today())
    if val is not None:
        return {
            "latest_available_date": today_str,
            "lag_days": 0,
            "note": f"Open-Meteo ERA5 data is available up to today ({today_str}). No lag.",
        }
    else:
        from datetime import timedelta
        yesterday = date.today() - timedelta(days=1)
        val2 = fetch_rainfall_only(datetime.combine(yesterday, datetime.min.time()))
        if val2 is not None:
            return {
                "latest_available_date": str(yesterday),
                "lag_days": 1,
                "note": f"Open-Meteo ERA5 data available up to {yesterday} (~1-day lag).",
            }
    return {
        "latest_available_date": "unknown",
        "lag_days": -1,
        "note": "Could not determine Open-Meteo data availability.",
    }
