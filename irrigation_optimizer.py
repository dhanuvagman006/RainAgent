import math
from typing import List, Dict, Any
from water_harvesting import TankStateTracker, calculate_harvestable_volume

CROP_REGISTRY = {
    "Black Pepper": {"kc": 0.80},
    "Banana": {"kc": 1.15},
    "Cocoa": {"kc": 0.95}
}

def calculate_eto(t_mean: float) -> float:
    """
    Approximates Daily Baseline Evapotranspiration (ETo) in mm based on mean temperature.
    Simple linear proxy for tropical/sub-tropical conditions.
    """
    return max(0.0, t_mean * 0.2)

def calculate_effective_rainfall(rainfall_mm: float) -> float:
    """
    USDA Soil Conservation Service method approximation for Effective Rainfall (P_eff).
    """
    if rainfall_mm <= 8.3:
        return rainfall_mm * 0.6
    else:
        return (rainfall_mm * 0.8) + 0.3

def generate_irrigation_plan(
    predicted_rainfall: List[float],
    simulated_temperatures: List[float],
    crop_name: str,
    catchment_area: float,
    cultivation_area: float,
    initial_tank_water: float,
    max_tank_capacity: float
) -> List[Dict[str, Any]]:
    """
    Generates a multi-day irrigation schedule optimizing tank water usage.
    """
    if crop_name not in CROP_REGISTRY:
        raise ValueError(f"Crop {crop_name} not found in registry.")
        
    kc = CROP_REGISTRY[crop_name]["kc"]
    tank = TankStateTracker(max_tank_capacity, initial_tank_water)
    
    schedule = []
    moisture_carry_over = 0.0 # surplus mm from previous day
    
    horizon = min(len(predicted_rainfall), len(simulated_temperatures))
    
    for day in range(horizon):
        rain = predicted_rainfall[day]
        t_mean = simulated_temperatures[day]
        
        # 1. Harvest Water
        harvested_liters = calculate_harvestable_volume(rain, catchment_area)
        tank.add_water(harvested_liters)
        
        # 2. Agronomic Calculations
        eto = calculate_eto(t_mean)
        etc = eto * kc
        peff = calculate_effective_rainfall(rain)
        
        # 3. Net Deficit with Dynamic Soil Moisture Buffer
        ir_mm = etc - peff - moisture_carry_over
        
        action = "No Irrigation"
        liters_needed = 0.0
        status_msg = "No Irrigation Needed: Rainfall or soil moisture meets crop demands."
        
        if ir_mm <= 0:
            # Surplus moisture
            surplus = abs(ir_mm)
            moisture_carry_over = surplus * 0.50 # carry forward 50% to next day
        else:
            # Deficit
            moisture_carry_over = 0.0
            liters_needed = ir_mm * cultivation_area # 1 mm deficit = 1 L / sqm
            
            # Draw from tank
            drawn = tank.draw_water(liters_needed)
            
            if drawn >= liters_needed:
                action = "Irrigate"
                status_msg = f"Sufficient tank water. Supplying {round(drawn, 2)} liters."
            else:
                action = "Irrigate (Partial)"
                status_msg = f"Deficit: Insufficient tank water. Supplying {round(drawn, 2)}/{round(liters_needed, 2)} liters. Top up required."
                liters_needed = drawn # The actual amount we are able to provide
        
        tank_status = tank.get_status()
        
        schedule.append({
            "day_index": day + 1,
            "predicted_rainfall_mm": round(rain, 2),
            "simulated_temperature_c": round(t_mean, 2),
            "harvested_water_liters": round(harvested_liters, 2),
            "crop_water_needed_liters": round(ir_mm * cultivation_area if ir_mm > 0 else 0, 2),
            "action": action,
            "status_message": status_msg,
            "tank_status_liters": tank_status["current_volume"],
            "tank_status_percentage": tank_status["percentage"]
        })
        
    return schedule
