"""
data_preprocessing.py  —  RainAgent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSE-maximising feature engineering (≥45 features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature groups added:
  1.  Cyclical encodings  — DOY, month, week-of-year (sin/cos)
  2.  Monsoon flags       — SW-monsoon, NE-monsoon, pre-season, dry
  3.  Extended lags       — 1,2,3,5,7,10,14,21,30 days
  4.  Rolling statistics  — mean/std/max over 3,7,14,30,60,90 days
  5.  Wet/dry streaks     — consecutive wet/dry day counters
  6.  Derived physics     — dewpoint, VPD, diurnal range,
                            pressure gradient, heat index proxy
  7.  Interaction terms   — rh×ws, solar×rh, monsoon×lag7
  8.  Log transforms      — log1p of lags to handle heavy tails
  9.  Cumulative monsoon  — running annual total (reset Jan 1)
 10.  Anomaly             — residual from 30-day rolling mean
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling(series, window, stat='mean', min_p=1):
    r = series.rolling(window, min_periods=min_p)
    if stat == 'mean':
        return r.mean()
    if stat == 'std':
        return r.std().fillna(0)
    if stat == 'max':
        return r.max()
    raise ValueError(stat)


def _wet_dry_streaks(rainfall_series, threshold=0.5):
    """Count consecutive wet and dry days up to (not including) current day."""
    vals = rainfall_series.values
    n = len(vals)
    wet_streak  = np.zeros(n, dtype=np.float32)
    dry_streak  = np.zeros(n, dtype=np.float32)
    ws = ds = 0
    for i in range(n):
        if i == 0:
            wet_streak[i] = dry_streak[i] = 0
        else:
            if vals[i - 1] >= threshold:
                ws += 1; ds = 0
            else:
                ds += 1; ws = 0
            wet_streak[i] = ws
            dry_streak[i] = ds
    return wet_streak, dry_streak


def _cumulative_season_rain(df_with_date, rainfall_col):
    """Running total of rainfall since Jan 1 of the current year."""
    cum = np.zeros(len(df_with_date), dtype=np.float32)
    running = 0.0
    prev_year = None
    for i, (yr, r) in enumerate(zip(df_with_date['_year_'], df_with_date[rainfall_col])):
        if yr != prev_year:
            running = 0.0
            prev_year = yr
        cum[i] = running
        running += r
    return cum


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean_data(file_path):
    """
    Load CSV, clean NaNs, and engineer ≥45 hydrologically-meaningful features.
    Returns a cleaned, feature-enriched DataFrame.
    """
    print(f"[DATA] Loading {file_path} ...")
    df = pd.read_csv(file_path)

    # Forward-fill then linear interpolate (preserves temporal ordering)
    df = df.ffill().interpolate(method='linear', limit_direction='both')

    # ── Date index ──────────────────────────────────────────────────────────
    date_col = None
    for candidate in ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp']:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

    # Stash year for cumulative rain helper
    df['_year_'] = df.index.year

    # ── 1. Cyclical time encodings ──────────────────────────────────────────
    doy   = df.index.day_of_year.values.astype(float)
    month = df.index.month.values.astype(float)
    week  = df.index.isocalendar().week.values.astype(float)

    df['doy_sin']   = np.sin(2 * np.pi * doy   / 365.25)
    df['doy_cos']   = np.cos(2 * np.pi * doy   / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * month  / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month  / 12.0)
    df['week_sin']  = np.sin(2 * np.pi * week   / 52.0)
    df['week_cos']  = np.cos(2 * np.pi * week   / 52.0)

    # ── 2. Monsoon window flags ─────────────────────────────────────────────
    m = df.index.month
    df['is_sw_monsoon'] = m.isin([6, 7, 8, 9]).astype(np.float32)
    df['is_ne_monsoon'] = m.isin([10, 11]).astype(np.float32)
    df['is_pre_season'] = m.isin([4, 5]).astype(np.float32)
    df['is_dry_season'] = m.isin([12, 1, 2, 3]).astype(np.float32)

    # ── 3. Rainfall lag features ────────────────────────────────────────────
    target = 'prectotcorr'
    if target in df.columns:
        for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
            df[f'rain_lag_{lag}d']     = df[target].shift(lag).fillna(0)
            # Log-transform of lag (handles heavy tail of rainfall distribution)
            df[f'rain_log_lag_{lag}d'] = np.log1p(df[f'rain_lag_{lag}d'])

        # ── 4. Rolling statistics ───────────────────────────────────────────
        for w in [3, 7, 14, 30, 60, 90]:
            df[f'rain_roll{w}_mean'] = _rolling(df[target], w, 'mean')
            df[f'rain_roll{w}_max']  = _rolling(df[target], w, 'max')
        for w in [3, 7, 14, 30]:
            df[f'rain_roll{w}_std']  = _rolling(df[target], w, 'std')

        # Wet-day fraction (proportion of days with rain > 0.5 mm)
        df['wet_day_frac7']  = (df[target] > 0.5).astype(float).rolling(7,  min_periods=1).mean()
        df['wet_day_frac30'] = (df[target] > 0.5).astype(float).rolling(30, min_periods=1).mean()
        df['wet_day_frac90'] = (df[target] > 0.5).astype(float).rolling(90, min_periods=1).mean()

        # Rainfall anomaly (deviation from 30-day baseline)
        df['rain_anomaly30'] = df[target] - df['rain_roll30_mean'].shift(1).fillna(0)

        # ── 5. Wet/dry streak counters ──────────────────────────────────────
        ws, ds = _wet_dry_streaks(df[target])
        df['wet_streak']  = ws.astype(np.float32)
        df['dry_streak']  = ds.astype(np.float32)

        # ── 9. Cumulative seasonal rainfall ─────────────────────────────────
        df['cum_year_rain'] = _cumulative_season_rain(df, target)

    # ── 6. Derived physics features ─────────────────────────────────────────
    if all(c in df.columns for c in ['t2m', 'rh2m']):
        # Magnus formula dewpoint (°C)
        a, b = 17.27, 237.7
        alpha = (a * df['t2m']) / (b + df['t2m']) + np.log(df['rh2m'] / 100.0)
        df['dewpoint'] = (b * alpha) / (a - alpha)

        # Vapour pressure deficit (hPa) — saturation VP minus actual VP
        es = 6.1078 * np.exp(a * df['t2m'] / (b + df['t2m']))   # sat VP
        ea = es * df['rh2m'] / 100.0                              # actual VP
        df['vpd'] = np.clip(es - ea, 0.0, None)

        # Wet-bulb proxy (Stull 2011 approximation)
        df['wet_bulb'] = (
            df['t2m'] * np.arctan(0.151977 * (df['rh2m'] + 8.313659) ** 0.5)
            + np.arctan(df['t2m'] + df['rh2m'])
            - np.arctan(df['rh2m'] - 1.676331)
            + 0.00391838 * df['rh2m'] ** 1.5 * np.arctan(0.023101 * df['rh2m'])
            - 4.686035
        )

    if all(c in df.columns for c in ['t2m_max', 't2m_min']):
        df['diurnal_range'] = df['t2m_max'] - df['t2m_min']

    if 'ps' in df.columns:
        df['pressure_gradient'] = df['ps'].diff().fillna(0)   # day-over-day change

    # ── 7. Interaction terms ────────────────────────────────────────────────
    if all(c in df.columns for c in ['rh2m', 'ws2m']):
        df['rh_ws_interaction'] = df['rh2m'] * df['ws2m']

    if all(c in df.columns for c in ['allsky_sfc_sw_dwn', 'rh2m']):
        df['solar_rh'] = df['allsky_sfc_sw_dwn'] * df['rh2m']

    if target in df.columns and 'is_sw_monsoon' in df.columns:
        df['monsoon_x_lag7'] = df['is_sw_monsoon'] * df.get('rain_lag_7d', 0.0)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    df = df.drop(columns=['_year_'], errors='ignore')
    df = df.reset_index(drop=True)
    df = df.fillna(0)      # safety net for any edge NaNs from rolling/lag

    print(f"[DATA] Shape after feature engineering: {df.shape}")
    print(f"[DATA] {len([c for c in df.columns if c != target])} features → target '{target}'")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scaling  (log1p + MinMax — unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def scale_data(df, target_col='prectotcorr', save_dir='models'):
    """Legacy wrapper → delegates to scale_data_log in train.py."""
    from train import scale_data_log
    return scale_data_log(df, target_col=target_col, save_dir=save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences(features, target, x_days=60, y_days=1):
    """
    Sliding-window sequence generator.

    Args:
        features : np.ndarray  (N, F)
        target   : np.ndarray  (N, 1)
        x_days   : lookback window length
        y_days   : forecast horizon

    Returns:
        X : (samples, x_days, F)
        y : (samples, y_days)
    """
    X, y = [], []
    num_samples = len(features) - x_days - y_days + 1

    for i in range(num_samples):
        X.append(features[i : i + x_days])
        y.append(target[i + x_days : i + x_days + y_days].flatten())

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[DATA] Sequences: X={X.shape}  y={y.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "dakshina_kannada_rainfall_synthetic.csv"
    df = load_and_clean_data(csv)
    print(df.head(3).to_string())
    print("\nColumn list:")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:>3}. {c}")
