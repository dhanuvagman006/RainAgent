"""
simulator.py  —  RainAgent
══════════════════════════
Generates synthetic weather data for inference.

CRITICAL: Must produce the EXACT same 23 features in the same order
that data_preprocessing.py engineers during training:
  year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m,
  allsky_sfc_sw_dwn, doy_sin, doy_cos, month_sin, month_cos,
  is_monsoon, rain_lag_1d, rain_lag_3d, rain_lag_7d,
  rain_roll7_mean, rain_roll30_mean, rain_roll7_std, wet_day_frac30
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def get_season(month):
    if 3 <= month <= 5:
        return 'summer'
    elif 6 <= month <= 9:
        return 'monsoon'
    else:
        return 'winter'


def generate_synthetic_weather(target_date_str, lookback_days=60, horizon=1):
    """
    Generates a realistic feature history based on the season.
    Produces (lookback_days + horizon - 1) days × 23 features,
    matching the exact feature set used during training.

    Args:
        target_date_str : 'YYYY-MM-DD'
        lookback_days   : sequence length (default 60, matches X_DAYS in train.py)
        horizon         : number of forecast days

    Returns:
        np.ndarray of shape (lookback_days + horizon - 1, 23)
    """
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    start_date = target_date - timedelta(days=lookback_days)

    total_days = lookback_days + horizon - 1
    dates = [start_date + timedelta(days=i) for i in range(total_days)]

    season = get_season(target_date.month)

    # ── Seasonal base profiles for raw meteorological columns ─────────────────
    profiles = {
        'summer':  {
            'ps': 99.0, 't2m': 30.0, 't2m_max': 35.0, 't2m_min': 25.0,
            'rh2m': 40.0, 'ws2m': 2.0, 'wd2m': 200.0, 'allsky_sfc_sw_dwn': 25.0,
            'rain_base': 0.5,   # typical daily rainfall (mm) for simulation
        },
        'monsoon': {
            'ps': 98.5, 't2m': 26.0, 't2m_max': 29.0, 't2m_min': 24.0,
            'rh2m': 85.0, 'ws2m': 4.0, 'wd2m': 250.0, 'allsky_sfc_sw_dwn': 10.0,
            'rain_base': 15.0,
        },
        'winter':  {
            'ps': 99.5, 't2m': 24.0, 't2m_max': 28.0, 't2m_min': 18.0,
            'rh2m': 60.0, 'ws2m': 1.5, 'wd2m': 150.0, 'allsky_sfc_sw_dwn': 20.0,
            'rain_base': 1.0,
        },
    }

    variance = {
        'ps': 0.2, 't2m': 1.5, 't2m_max': 1.5, 't2m_min': 1.5,
        'rh2m': 5.0, 'ws2m': 0.5, 'wd2m': 30.0, 'allsky_sfc_sw_dwn': 3.0,
    }

    base = profiles[season]

    # ── Step 1: generate raw meteorological rows + synthetic rainfall ─────────
    rows = []
    for d in dates:
        # Raw rainfall for lag/rolling engineering — Gamma-distributed
        # (heavy tail, never negative, realistic for tropical rainfall)
        rain_val = np.random.gamma(shape=0.8, scale=base['rain_base'])

        row = {
            'year':              d.year,
            'month':             d.month,
            'day':               d.day,
            'ps':                np.random.normal(base['ps'],               variance['ps']),
            't2m':               np.random.normal(base['t2m'],              variance['t2m']),
            't2m_max':           np.random.normal(base['t2m_max'],          variance['t2m_max']),
            't2m_min':           np.random.normal(base['t2m_min'],          variance['t2m_min']),
            'rh2m':              np.random.normal(base['rh2m'],             variance['rh2m']),
            'ws2m':              max(0.0, np.random.normal(base['ws2m'],    variance['ws2m'])),
            'wd2m':              np.random.normal(base['wd2m'],             variance['wd2m']) % 360,
            'allsky_sfc_sw_dwn': max(0.0, np.random.normal(base['allsky_sfc_sw_dwn'],
                                                             variance['allsky_sfc_sw_dwn'])),
            'prectotcorr':       rain_val,   # used for lag/rolling engineering
            'date':              d,          # needed for time-based features
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('date')

    # ── Step 2: engineer the SAME features as data_preprocessing.py ──────────
    from data_preprocessing import engineer_features
    import json
    import os

    df = engineer_features(df, target='prectotcorr')

    # ── Step 3: select EXACTLY the training features, in training order ────
    meta_path = os.path.join("models", "scaler_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta["feature_cols"]
    else:
        # Fallback just in case
        feature_cols = [c for c in df.columns if c != 'prectotcorr']

    return df[feature_cols].values  # shape: (total_days, num_features)
