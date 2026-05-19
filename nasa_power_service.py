"""
nasa_power_service.py
─────────────────────────────────────────────────────────────────────────────
Fetches real-time / historical meteorological data from the NASA POWER
Daily API for the Dakshina Kannada (Mangaluru) region.

Endpoint docs: https://power.larc.nasa.gov/docs/services/api/

Notes:
  - NASA POWER has a ~3-day processing lag. Data for the last 3 days
    will return the fill value (-999.0) and is treated as unavailable.
  - Station coordinates: Lat 12.87, Lon 74.88 (Dakshina Kannada, Karnataka)
  - All parameters match the training CSV columns exactly.
"""

import requests
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Station constants ──────────────────────────────────────────────────────
LATITUDE  = 12.87
LONGITUDE = 74.88
FILL_VALUE = -999.0

# NASA POWER parameters → CSV column mapping
PARAM_MAP = {
    "PRECTOTCORR":     "prectotcorr",
    "PS":              "ps",
    "T2M":             "t2m",
    "T2M_MAX":         "t2m_max",
    "T2M_MIN":         "t2m_min",
    "RH2M":            "rh2m",
    "WS2M":            "ws2m",
    "WD2M":            "wd2m",
    "ALLSKY_SFC_SW_DWN": "allsky_sfc_sw_dwn",
}

NASA_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

COMMUNITY = "AG"   # Agroclimatology community gives full parameter set


def _build_date_key(dt: datetime) -> str:
    """Convert datetime → YYYYMMDD string for NASA API."""
    return dt.strftime("%Y%m%d")


def fetch_daily_data(date: datetime) -> Optional[dict]:
    """
    Fetch a single day's data from NASA POWER for Dakshina Kannada.

    Returns a dict with CSV-compatible column names, e.g.:
        {
            'prectotcorr': 12.5,
            'ps':          98.7,
            't2m':         28.3,
            ...
        }
    Returns None if:
        - The date is in the future
        - NASA POWER has not yet processed the date (fill value = -999)
        - A network/API error occurs
    """
    today = datetime.utcnow().date()
    if date.date() > today:
        logger.info("Requested date %s is in the future — no data.", date.date())
        return None

    date_key = _build_date_key(date)
    params = {
        "parameters": ",".join(PARAM_MAP.keys()),
        "community":  COMMUNITY,
        "longitude":  LONGITUDE,
        "latitude":   LATITUDE,
        "start":      date_key,
        "end":        date_key,
        "format":     "JSON",
    }

    try:
        resp = requests.get(NASA_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.warning("NASA POWER API timed out for date %s", date_key)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("NASA POWER API request failed: %s", exc)
        return None
    except ValueError:
        logger.warning("NASA POWER API returned invalid JSON.")
        return None

    try:
        param_data = data["properties"]["parameter"]
    except (KeyError, TypeError):
        logger.warning("Unexpected NASA POWER response structure.")
        return None

    result = {}
    for nasa_key, csv_col in PARAM_MAP.items():
        raw_val = param_data.get(nasa_key, {}).get(date_key)
        if raw_val is None or raw_val == FILL_VALUE:
            logger.info(
                "Parameter %s for %s is missing or fill-value (-999) — "
                "data not yet processed by NASA.", nasa_key, date_key
            )
            return None          # Any missing param = whole record unavailable
        result[csv_col] = round(float(raw_val), 4)

    return result


def fetch_rainfall_only(date: datetime) -> Optional[float]:
    """
    Lightweight variant — fetches only the PRECTOTCORR (rainfall) value.
    Returns the rainfall in mm/day, or None if unavailable.
    """
    today = datetime.utcnow().date()
    if date.date() > today:
        return None

    date_key = _build_date_key(date)
    params = {
        "parameters": "PRECTOTCORR",
        "community":  COMMUNITY,
        "longitude":  LONGITUDE,
        "latitude":   LATITUDE,
        "start":      date_key,
        "end":        date_key,
        "format":     "JSON",
    }

    try:
        resp = requests.get(NASA_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        val  = data["properties"]["parameter"]["PRECTOTCORR"].get(date_key)
        if val is None or val == FILL_VALUE:
            return None
        return round(float(val), 2)
    except Exception as exc:
        logger.warning("NASA POWER lightweight fetch failed: %s", exc)
        return None


def get_data_lag_info() -> dict:
    """
    Returns metadata about the NASA POWER processing lag:
        - latest_available_date (approx, ~3 days before today)
        - lag_days
    """
    from datetime import timedelta
    today = datetime.utcnow().date()
    # Try last 7 days to find latest available date
    for lag in range(1, 8):
        candidate = today - timedelta(days=lag)
        date_key  = candidate.strftime("%Y%m%d")
        try:
            resp = requests.get(
                NASA_API_URL,
                params={
                    "parameters": "PRECTOTCORR",
                    "community":  COMMUNITY,
                    "longitude":  LONGITUDE,
                    "latitude":   LATITUDE,
                    "start":      date_key,
                    "end":        date_key,
                    "format":     "JSON",
                },
                timeout=10,
            )
            val = resp.json()["properties"]["parameter"]["PRECTOTCORR"].get(date_key)
            if val is not None and val != FILL_VALUE:
                return {
                    "latest_available_date": str(candidate),
                    "lag_days": lag,
                    "note": f"NASA POWER data is currently available up to {candidate} ({lag}-day lag).",
                }
        except Exception:
            continue
    return {
        "latest_available_date": "unknown",
        "lag_days": -1,
        "note": "Could not determine NASA POWER data lag.",
    }
