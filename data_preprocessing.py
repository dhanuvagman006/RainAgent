"""
data_preprocessing.py  —  RainAgent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enhanced feature engineering to maximise NSE:
  • Cyclical day-of-year encoding (sin/cos)  →  model learns seasonality
  • Rolling statistics (7-day / 30-day mean & std)  →  local trend signal
  • Lag features (1, 3, 7-day rainfall lags)  →  autocorrelation signal
  • Boolean monsoon-window flag               →  hard prior for Dakshina Kannada
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# Load & clean
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean_data(file_path):
    """
    Load CSV, clean NaNs, and engineer hydrologically-meaningful features.
    Returns a cleaned, feature-enriched DataFrame.
    """
    print(f"[DATA] Loading {file_path} ...")
    df = pd.read_csv(file_path)

    # Forward-fill then linear interpolate (preserves temporal ordering)
    df = df.ffill().interpolate(method='linear', limit_direction='both')

    # ── Date index (required for temporal feature engineering) ──────────────
    # Try common column names; if none exist, synthesise a date range
    date_col = None
    for candidate in ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp']:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        # Assume daily data starting 2000-01-01 when no date column found
        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

    # ── Cyclical day-of-year (captures monsoon seasonality perfectly) ────────
    doy = df.index.day_of_year.values.astype(float)
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    # ── Month cyclical encoding ──────────────────────────────────────────────
    month = df.index.month.values.astype(float)
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    # ── Monsoon window flag (Dakshina Kannada: June–September) ──────────────
    df['is_monsoon'] = df.index.month.isin([6, 7, 8, 9]).astype(np.float32)

    # ── Rainfall lag features (autocorrelation signal for wet spells) ────────
    target = 'prectotcorr'
    if target in df.columns:
        for lag in [1, 3, 7]:
            df[f'rain_lag_{lag}d'] = df[target].shift(lag).fillna(0)

        # Rolling statistics — short-window trend + variance signal
        df['rain_roll7_mean'] = (
            df[target].rolling(7,  min_periods=1).mean()
        )
        df['rain_roll30_mean'] = (
            df[target].rolling(30, min_periods=1).mean()
        )
        df['rain_roll7_std'] = (
            df[target].rolling(7,  min_periods=1).std().fillna(0)
        )

        # Wet-day fraction over last 30 days (proportion of days with rain > 1 mm)
        df['wet_day_frac30'] = (
            (df[target] > 1.0).astype(float)
            .rolling(30, min_periods=1).mean()
        )

    # Reset integer index for downstream compatibility
    df = df.reset_index(drop=True)
    df = df.fillna(0)   # safety net for any edge NaNs from rolling/lag

    print(f"[DATA] Shape after feature engineering: {df.shape}")
    print(f"[DATA] Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scaling  (log1p + MinMax — unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def scale_data(df, target_col='prectotcorr', save_dir='models'):
    """Legacy scale_data — wraps scale_data_log for backward compatibility."""
    from train import scale_data_log   # avoid circular import at module level
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
    df = load_and_clean_data("dakshina_kannada_rainfall_daily_2000_2024.csv")
    print(df.head())
