from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from simulator import generate_synthetic_weather
from inference_service import inference_service
from irrigation_optimizer import generate_irrigation_plan

# ── Pre-load the historical dataset once at startup ──────────────────────────
_CSV_PATH = "dakshina_kannada_rainfall_daily_2000_2024.csv"
try:
    _HISTORICAL_DF = pd.read_csv(_CSV_PATH).ffill().interpolate(
        method='linear', limit_direction='both'
    )
except Exception:
    _HISTORICAL_DF = None

app = FastAPI(title="Smart Rainfall Prediction API", description="Module 2 Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    date: str = Field(..., description="Target prediction date in YYYY-MM-DD")
    model_name: str = Field(..., description="LSTM, GRU, Bi-LSTM, 1D-CNN, CNN-LSTM, Transformer, or Ensemble")
    horizon: int = Field(1, description="Number of forecast days (1, 4, 7)")

class PredictionResponse(BaseModel):
    dates: List[str]
    predicted_rainfall_mm: List[float]
    simulated_weather_summary: Dict[str, float]

class IrrigationRequest(BaseModel):
    predicted_rainfall: List[float]
    simulated_temperatures: List[float]
    crop_name: str
    catchment_area: float
    cultivation_area: float
    initial_tank_water: float
    max_tank_capacity: float

class IrrigationResponse(BaseModel):
    schedule: List[Dict[str, Any]]

@app.get("/metrics")
def get_metrics():
    metrics_path = "training_metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Training metrics not found. Run Module 1 first.")
    
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data


# ── Validation response schema ────────────────────────────────────────────────
class ValidationResponse(BaseModel):
    date:                   str
    imd_actual_rainfall_mm: Optional[float]
    data_available:         bool
    station:                str
    source:                 str
    dataset_range:          str
    note:                   str

@app.get("/validate-actual-rainfall", response_model=ValidationResponse)
def validate_actual_rainfall(date: str):
    """
    Looks up the historically observed rainfall for a given date from the
    NASA POWER / IMD-aligned Dakshina Kannada dataset (2000-2024).
    Returns the actual prectotcorr value as ground-truth.
    """
    try:
        target = datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if _HISTORICAL_DF is None:
        raise HTTPException(status_code=503, detail="Historical dataset not available.")

    # Filter to the requested date
    row = _HISTORICAL_DF[
        (_HISTORICAL_DF['year']  == target.year) &
        (_HISTORICAL_DF['month'] == target.month) &
        (_HISTORICAL_DF['day']   == target.day)
    ]

    if row.empty:
        return ValidationResponse(
            date=date,
            imd_actual_rainfall_mm=None,
            data_available=False,
            station="Dakshina Kannada Region (Mangaluru)",
            source="India Meteorological Department (IMD) · NASA POWER",
            dataset_range="2000-01-01 to 2024-12-31",
            note="Date is outside the available observation window (2000–2024). "
                 "Future dates have no ground-truth record yet.",
        )

    actual_mm = round(float(row['prectotcorr'].iloc[0]), 2)
    return ValidationResponse(
        date=date,
        imd_actual_rainfall_mm=actual_mm,
        data_available=True,
        station="Dakshina Kannada Region (Mangaluru)",
        source="India Meteorological Department (IMD) · NASA POWER",
        dataset_range="2000-01-01 to 2024-12-31",
        note="Ground-truth daily rainfall from the NASA POWER / IMD station network "
             "co-located at Dakshina Kannada, Karnataka.",
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_rainfall(request: PredictionRequest):
    try:
        target_date = datetime.strptime(request.date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
    valid_horizons = [1, 4, 7]
    if request.horizon not in valid_horizons:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {valid_horizons}.")
        
    valid_models = ["LSTM", "GRU", "Bi-LSTM", "1D-CNN", "CNN-LSTM", "Transformer", "Ensemble"]
    if request.model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Model must be one of {valid_models}.")

    # 1. Generate synthetic features
    try:
        synthetic_data = generate_synthetic_weather(request.date, lookback_days=30, horizon=request.horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")
        
    # 2. Run Inference
    try:
        predictions = inference_service.predict(request.model_name, synthetic_data, request.horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        
    # 3. Format Response
    dates = [(target_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(request.horizon)]
    
    last_day_data = synthetic_data[29]
    weather_summary = {
        "ps": round(float(last_day_data[3]), 2),
        "t2m": round(float(last_day_data[4]), 2),
        "t2m_max": round(float(last_day_data[5]), 2),
        "t2m_min": round(float(last_day_data[6]), 2),
        "rh2m": round(float(last_day_data[7]), 2),
        "ws2m": round(float(last_day_data[8]), 2),
        "allsky_sfc_sw_dwn": round(float(last_day_data[10]), 2),
    }

    return PredictionResponse(
        dates=dates,
        predicted_rainfall_mm=[round(float(p), 2) for p in predictions],
        simulated_weather_summary=weather_summary
    )

@app.post("/calculate-irrigation-plan", response_model=IrrigationResponse)
def calculate_irrigation(request: IrrigationRequest):
    try:
        schedule = generate_irrigation_plan(
            predicted_rainfall=request.predicted_rainfall,
            simulated_temperatures=request.simulated_temperatures,
            crop_name=request.crop_name,
            catchment_area=request.catchment_area,
            cultivation_area=request.cultivation_area,
            initial_tank_water=request.initial_tank_water,
            max_tank_capacity=request.max_tank_capacity
        )
        return IrrigationResponse(schedule=schedule)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Irrigation calculation failed: {e}")
