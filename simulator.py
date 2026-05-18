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

def generate_synthetic_weather(target_date_str, lookback_days=30, horizon=1):
    """
    Generates a realistic feature history based on the season.
    Target date format: YYYY-MM-DD
    Generates (lookback_days + horizon - 1) days of data.
    """
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    start_date = target_date - timedelta(days=lookback_days)
    
    total_days = lookback_days + horizon - 1
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    
    season = get_season(target_date.month)
    
    # Base profiles for meteorological columns
    # Columns: ['ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
    profiles = {
        'summer': {
            'ps': 99.0, 't2m': 30.0, 't2m_max': 35.0, 't2m_min': 25.0,
            'rh2m': 40.0, 'ws2m': 2.0, 'wd2m': 200.0, 'allsky_sfc_sw_dwn': 25.0
        },
        'monsoon': {
            'ps': 98.5, 't2m': 26.0, 't2m_max': 29.0, 't2m_min': 24.0,
            'rh2m': 85.0, 'ws2m': 4.0, 'wd2m': 250.0, 'allsky_sfc_sw_dwn': 10.0
        },
        'winter': {
            'ps': 99.5, 't2m': 24.0, 't2m_max': 28.0, 't2m_min': 18.0,
            'rh2m': 60.0, 'ws2m': 1.5, 'wd2m': 150.0, 'allsky_sfc_sw_dwn': 20.0
        }
    }
    
    # Define variance (standard deviation for Gaussian noise)
    variance = {
        'ps': 0.2, 't2m': 1.5, 't2m_max': 1.5, 't2m_min': 1.5,
        'rh2m': 5.0, 'ws2m': 0.5, 'wd2m': 30.0, 'allsky_sfc_sw_dwn': 3.0
    }
    
    base = profiles[season]
    
    synthetic_data = []
    
    for d in dates:
        row = {
            'year': d.year,
            'month': d.month,
            'day': d.day,
            'ps': np.random.normal(base['ps'], variance['ps']),
            't2m': np.random.normal(base['t2m'], variance['t2m']),
            't2m_max': np.random.normal(base['t2m_max'], variance['t2m_max']),
            't2m_min': np.random.normal(base['t2m_min'], variance['t2m_min']),
            'rh2m': np.random.normal(base['rh2m'], variance['rh2m']),
            'ws2m': max(0, np.random.normal(base['ws2m'], variance['ws2m'])),
            'wd2m': np.random.normal(base['wd2m'], variance['wd2m']) % 360,
            'allsky_sfc_sw_dwn': max(0, np.random.normal(base['allsky_sfc_sw_dwn'], variance['allsky_sfc_sw_dwn']))
        }
        synthetic_data.append(row)
        
    df = pd.DataFrame(synthetic_data)
    
    # Ensure columns match training order precisely
    feature_cols = ['year', 'month', 'day', 'ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
    df = df[feature_cols]
    
    return df.values
