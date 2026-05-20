"""
download_real_dataset.py  —  RainAgent
══════════════════════════════════════════════════════════════════════════════
Downloads REAL-WORLD daily meteorological data from NASA POWER for the
Dakshina Kannada (Mangaluru) region for the period 2020-01-01 → 2024-12-31.

Strategy:
  • NASA POWER bulk API supports up to 366-day windows per request.
  • We split the 5-year range into yearly chunks (≤366 days each) to stay
    well within API limits and get fast, reliable responses.
  • Each year is fetched in one HTTP call; results are merged into a single CSV.

Output:
  dakshina_kannada_rainfall_real.csv

Usage:
  python download_real_dataset.py
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import requests
import pandas as pd
from datetime import datetime, date

# ── Station / API constants ────────────────────────────────────────────────
LATITUDE   = 12.87
LONGITUDE  = 74.88
COMMUNITY  = "AG"
FILL_VALUE = -999.0
NASA_API   = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMETERS = [
    "PRECTOTCORR",       # Rainfall (mm/day)
    "PS",                # Surface pressure (kPa)
    "T2M",               # Temperature at 2 m (°C)
    "T2M_MAX",           # Max temperature at 2 m (°C)
    "T2M_MIN",           # Min temperature at 2 m (°C)
    "RH2M",              # Relative humidity at 2 m (%)
    "WS2M",              # Wind speed at 2 m (m/s)
    "WD2M",              # Wind direction at 2 m (°)
    "ALLSKY_SFC_SW_DWN", # All-sky surface solar radiation (MJ/m²/day)
]

# CSV column names (matching the rest of the project)
COL_MAP = {
    "PRECTOTCORR":       "prectotcorr",
    "PS":                "ps",
    "T2M":               "t2m",
    "T2M_MAX":           "t2m_max",
    "T2M_MIN":           "t2m_min",
    "RH2M":              "rh2m",
    "WS2M":              "ws2m",
    "WD2M":              "wd2m",
    "ALLSKY_SFC_SW_DWN": "allsky_sfc_sw_dwn",
}

START_YEAR = 2020
END_YEAR   = 2025
OUT_FILE   = "dakshina_kannada_rainfall_real.csv"


# ── Helpers ────────────────────────────────────────────────────────────────

def _clamp_end(year: int) -> date:
    """Return end date for a year, capped at today-3 (NASA lag)."""
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=3)
    dec31  = date(year, 12, 31)
    return min(dec31, cutoff)


def fetch_year(year: int) -> pd.DataFrame:
    """
    Fetch a full calendar year of data from NASA POWER in one API call.
    Returns a DataFrame with columns: year, month, day, <met vars>.
    """
    start = date(year, 1, 1)
    end   = _clamp_end(year)

    if end < start:
        print(f"  ⚠  Year {year}: no data available yet (within NASA lag window).")
        return pd.DataFrame()

    start_str = start.strftime("%Y%m%d")
    end_str   = end.strftime("%Y%m%d")

    print(f"  → Fetching {year}  ({start_str} – {end_str}) … ", end="", flush=True)

    params = {
        "parameters": ",".join(PARAMETERS),
        "community":  COMMUNITY,
        "longitude":  LONGITUDE,
        "latitude":   LATITUDE,
        "start":      start_str,
        "end":        end_str,
        "format":     "JSON",
    }

    for attempt in range(1, 4):          # up to 3 retries
        try:
            resp = requests.get(NASA_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.Timeout:
            print(f"\n  ⚠  Timeout (attempt {attempt}/3). Retrying …")
            time.sleep(5 * attempt)
        except requests.exceptions.HTTPError as exc:
            print(f"\n  ✗  HTTP error {exc}. Skipping year {year}.")
            return pd.DataFrame()
        except Exception as exc:
            print(f"\n  ✗  Unexpected error: {exc}. Skipping year {year}.")
            return pd.DataFrame()
    else:
        print(f"\n  ✗  All retries failed for year {year}. Skipping.")
        return pd.DataFrame()

    # ── Parse response ───────────────────────────────────────────────────
    try:
        param_data = data["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        print(f"\n  ✗  Unexpected API response structure: {exc}")
        return pd.DataFrame()

    # Build a dict: date_string → {col: value}
    date_rows: dict = {}

    for nasa_key in PARAMETERS:
        col = COL_MAP[nasa_key]
        daily = param_data.get(nasa_key, {})
        for date_str, val in daily.items():
            if date_rows.get(date_str) is None:
                date_rows[date_str] = {}
            date_rows[date_str][col] = None if (val is None or val == FILL_VALUE) else round(float(val), 4)

    rows = []
    for date_str in sorted(date_rows.keys()):
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue
        row = {
            "year":  dt.year,
            "month": dt.month,
            "day":   dt.day,
        }
        row.update(date_rows[date_str])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows parsed.")
        return df

    # Count fill-value (None) cells
    missing = df.isnull().sum().sum()
    print(f"✔  {len(df)} days  (missing cells: {missing})")
    return df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NASA POWER Real-World Dataset Downloader — Dakshina Kannada ║")
    print(f"║  Period : {START_YEAR}-01-01  →  {END_YEAR}-12-31               ║")
    print(f"║  Coords : Lat {LATITUDE}°N  Lon {LONGITUDE}°E                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        df = fetch_year(year)
        if not df.empty:
            frames.append(df)
        time.sleep(1)   # polite pause between requests

    if not frames:
        print("\n✗  No data was downloaded. Check your internet connection.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # ── Ensure correct column order ──────────────────────────────────────
    ordered_cols = (
        ["year", "month", "day"]
        + [COL_MAP[p] for p in PARAMETERS]
    )
    combined = combined[[c for c in ordered_cols if c in combined.columns]]

    # ── Handle remaining fill values (NaN → forward-fill then back-fill) ─
    met_cols = [c for c in combined.columns if c not in ("year", "month", "day")]
    n_before = combined[met_cols].isnull().sum().sum()
    combined[met_cols] = (
        combined[met_cols]
        .ffill()
        .bfill()
    )
    n_after = combined[met_cols].isnull().sum().sum()
    if n_before:
        print(f"\n  ℹ  Filled {n_before} missing cells via forward/back-fill "
              f"(remaining: {n_after}).")

    # ── Save ─────────────────────────────────────────────────────────────
    combined.to_csv(OUT_FILE, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    total_days  = len(combined)
    total_years = total_days / 365.25
    annual_rain = combined["prectotcorr"].sum() / total_years if total_years else 0
    wet_days    = (combined["prectotcorr"] > 0.5).sum()
    max_rain    = combined["prectotcorr"].max()

    print(f"\n{'─'*60}")
    print(f"  ✔  Real dataset saved → {OUT_FILE}")
    print(f"{'─'*60}")
    print(f"  Rows          : {total_days:,}  ({START_YEAR}–{END_YEAR})")
    print(f"  Annual mean   : {annual_rain:.0f} mm/yr  (expected DK ≈ 3 800–4 200 mm)")
    print(f"  Wet days      : {wet_days:,}  ({100 * wet_days / total_days:.1f}%)")
    print(f"  Max daily     : {max_rain:.1f} mm")
    print(f"\n  Column summary:")
    print(combined[met_cols].describe().round(2).to_string())
    print(f"{'─'*60}")
    print("\nDone! You can now use this CSV to retrain the models with real data.\n")


if __name__ == "__main__":
    main()
