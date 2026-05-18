class TankStateTracker:
    def __init__(self, tank_capacity: float, initial_volume: float):
        self.tank_capacity = tank_capacity
        self.current_volume = min(max(initial_volume, 0.0), tank_capacity)

    def add_water(self, amount: float):
        """Adds water to the tank, capping at max capacity."""
        if amount < 0:
            return
        self.current_volume = min(self.current_volume + amount, self.tank_capacity)

    def draw_water(self, amount: float) -> float:
        """
        Draws water from the tank. 
        Returns the actual amount drawn (may be less than requested if tank is empty).
        """
        if amount <= 0:
            return 0.0
        
        if self.current_volume >= amount:
            self.current_volume -= amount
            return amount
        else:
            drawn = self.current_volume
            self.current_volume = 0.0
            return drawn

    def get_status(self) -> dict:
        percentage = (self.current_volume / self.tank_capacity) * 100 if self.tank_capacity > 0 else 0
        return {
            "current_volume": round(self.current_volume, 2),
            "percentage": round(percentage, 2)
        }

def calculate_harvestable_volume(rainfall_mm: float, catchment_area_sqm: float, runoff_coefficient: float = 0.85) -> float:
    """
    Calculates harvestable water in Liters using the Rational Method.
    1 mm of rain over 1 sqm = 1 Liter.
    """
    return max(0.0, rainfall_mm * catchment_area_sqm * runoff_coefficient)
