from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

from simulator import generate_synthetic_weather
from inference_service import inference_service
from irrigation_optimizer import generate_irrigation_plan

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
