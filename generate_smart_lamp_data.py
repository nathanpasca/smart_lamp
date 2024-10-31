import random
import time
import csv
from datetime import datetime, timedelta
import math

class IndonesianPowerRates:
    """Indonesian Power Rate Calculator based on PLN rates"""
    
    def __init__(self):
        # PLN rates in IDR/kWh for different categories (2024 rates)
        self.rates = {
            'R1': {  # Residential 900VA
                'base_rate': 1352,
                'peak_multiplier': 1.5,  # During peak hours
                'peak_hours': [17, 18, 19, 20, 21, 22]  # 17:00-22:00
            },
            'R2': {  # Residential 1300VA-2200VA
                'base_rate': 1444.70,
                'peak_multiplier': 1.5,
                'peak_hours': [17, 18, 19, 20, 21, 22]
            },
            'B1': {  # Business 1300VA-2200VA
                'base_rate': 1444.70,
                'peak_multiplier': 1.5,
                'peak_hours': [9, 10, 11, 12, 13, 14, 15, 16]  # Business hours
            }
        }

class IndonesianSmartLampSensor:
    def __init__(self, power_category='R1'):
        self.ambient_light_base = 0
        self.room_occupied = False
        self.current_brightness = 0
        self.current_color_temp = 2000
        self.power_rates = IndonesianPowerRates()
        self.power_category = power_category
        
        # Indonesia-specific occupancy patterns
        self.occupancy_probability = {
            # Hour: (weekday_prob, weekend_prob)
            0: (0.1, 0.2),    # Very low late night
            1: (0.1, 0.15),
            2: (0.1, 0.1),
            3: (0.1, 0.1),
            4: (0.2, 0.15),   # Early morning prayer time
            5: (0.3, 0.2),    # Subuh prayer & morning activities
            6: (0.4, 0.3),    # Morning routine
            7: (0.7, 0.4),    # Commute time
            8: (0.9, 0.5),    # Work starts
            9: (0.9, 0.6),
            10: (0.9, 0.7),
            11: (0.8, 0.7),   # Approaching Dzuhur prayer
            12: (0.7, 0.8),   # Lunch & prayer time
            13: (0.9, 0.7),
            14: (0.9, 0.6),
            15: (0.8, 0.6),   # Ashar prayer time
            16: (0.8, 0.5),
            17: (0.7, 0.5),   # End of work day
            18: (0.6, 0.6),   # Maghrib prayer time
            19: (0.5, 0.7),   # Evening activities
            20: (0.4, 0.6),   # Isha prayer time
            21: (0.3, 0.5),
            22: (0.2, 0.3),
            23: (0.1, 0.2)
        }

    def calculate_optimal_color_temp(self, hour, occupied):
        """Calculate optimal color temperature based on time of day in Indonesia"""
        if not occupied:
            return 2000
        
        # Color temperature adjustments for Indonesian daylight patterns
        if 4 <= hour < 6:    # Subuh prayer and early morning
            return random.uniform(4000, 5000)  # Gentle wake-up light
        elif 6 <= hour < 9:  # Morning
            return random.uniform(5000, 6500)  # Energizing cool light
        elif 9 <= hour < 16: # Working hours
            return random.uniform(4200, 5000)  # Productive neutral light
        elif 16 <= hour < 18:# Late afternoon
            return random.uniform(3500, 4200)  # Transition period
        elif hour >= 18:     # Evening/night (after Maghrib)
            return random.uniform(2000, 3000)  # Warm, relaxing light
        else:               # Late night/early morning
            return random.uniform(2000, 2500)  # Very warm light

    def calculate_power_cost(self, energy_consumption, hour):
        """Calculate power cost based on PLN rates"""
        rates = self.power_rates.rates[self.power_category]
        
        # Check if current hour is during peak time
        is_peak_hour = hour in rates['peak_hours']
        
        # Calculate rate with peak hour consideration
        current_rate = rates['base_rate'] * (rates['peak_multiplier'] if is_peak_hour else 1)
        
        # Convert kWh to IDR
        cost_idr = energy_consumption * current_rate
        
        return cost_idr

    def simulate_ambient_light(self, hour):
        """Simulate realistic ambient light levels based on Indonesian daylight patterns"""
        # Indonesia typically has sunrise ~5:30-6:00 and sunset ~17:30-18:00
        if 5.5 <= hour <= 18:
            # Daylight hours - stronger tropical sun
            hour_normalized = (hour - 5.5) / 12.5  # 0 to 1 over daylight hours
            
            # Higher base light levels due to tropical location
            base_light = 1200 * math.exp(-(pow(hour_normalized - 0.5, 2) / 0.08))
            
            # Add random variations for cloud cover (common in tropical climate)
            cloud_factor = random.uniform(0.6, 1.0)
            light_level = base_light * cloud_factor
            
            # Add random variations for seasonal changes (less pronounced near equator)
            seasonal_variation = random.uniform(0.9, 1.1)
            light_level *= seasonal_variation
        else:
            # Night hours - urban light pollution consideration
            light_level = random.uniform(5, 40)  # Higher min due to urban areas
        
        return max(0, min(1000, light_level))

    def simulate_occupancy(self, hour, is_weekend):
        """Simulate room occupancy based on Indonesian work patterns"""
        weekday_prob, weekend_prob = self.occupancy_probability[hour]
        prob = weekend_prob if is_weekend else weekday_prob
        
        # Prayer time adjustments
        prayer_times = [5, 12, 15, 18, 19]  # Approximate prayer times
        if hour in prayer_times:
            prob *= 0.7  # Reduced occupancy during prayer times
            
        return random.random() < prob

    def calculate_optimal_brightness(self, ambient_light, occupied, hour):
        """Calculate optimal lamp brightness based on conditions and time"""
        if not occupied:
            return 0
        
        # Adjusted for tropical lighting conditions and power saving during peak hours
        rates = self.power_rates.rates[self.power_category]
        is_peak_hour = hour in rates['peak_hours']
        
        # Reduce brightness during peak hours to save power
        peak_reduction = 0.8 if is_peak_hour else 1.0
        
        if ambient_light > 600:  # Higher threshold for tropical sunlight
            brightness = random.uniform(15, 35) * peak_reduction
        elif ambient_light > 300:
            brightness = random.uniform(35, 65) * peak_reduction
        else:
            brightness = random.uniform(65, 100) * peak_reduction
            
        return max(0, min(100, brightness))

    def calculate_energy_consumption(self, brightness, hour, duration_hours=1):
        """Calculate energy consumption based on brightness and local power factors"""
        # Base consumption for 220V system (in kWh)
        base_consumption = 0.012  # Typical LED smart bulb at 220V
        
        # Calculate actual consumption
        energy = (brightness / 100) * base_consumption * duration_hours
        
        return energy

    def generate_reading(self, timestamp):
        """Generate a single sensor reading"""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5  # Saturday and Sunday
        
        # Generate sensor values
        ambient_light = self.simulate_ambient_light(hour)
        occupied = self.simulate_occupancy(hour, is_weekend)
        
        if occupied:
            brightness = self.calculate_optimal_brightness(ambient_light, occupied, hour)
            color_temp = self.calculate_optimal_color_temp(hour, occupied)
        else:
            brightness = 0
            color_temp = 2000
            
        energy = self.calculate_energy_consumption(brightness, hour)
        power_cost = self.calculate_power_cost(energy, hour)
        
        return {
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Day_of_Week': timestamp.weekday(),
            'Hour_of_Day': hour,
            'Ambient_Light_Level': round(ambient_light, 2),
            'Room_Occupied': int(occupied),
            'Current_Lamp_Brightness': round(brightness, 2),
            'Current_Lamp_Color_Temp': round(color_temp, 2),
            'Energy_Consumption_kWh': round(energy, 4),
            'Power_Cost_IDR': round(power_cost, 2)
        }

def generate_dataset(days=7, interval_minutes=45, power_category='R1'):
    """Generate a dataset covering specified number of days"""
    sensor = IndonesianSmartLampSensor(power_category)
    data = []
    
    # Start from current date
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Generate readings
    current_time = start_time
    end_time = start_time + timedelta(days=days)
    
    while current_time < end_time:
        reading = sensor.generate_reading(current_time)
        data.append(reading)
        current_time += timedelta(minutes=interval_minutes)
    
    return data

def save_to_csv(data, filename='smart_lamp_sensor_data_indonesia.csv'):
    """Save generated data to CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# Example usage
if __name__ == "__main__":
    print("Generating Indonesian smart lamp sensor data with power consumption...")
    # Generate data for different power categories
    for category in ['R1', 'R2', 'B1']:
        data = generate_dataset(days=30, power_category=category)
        save_to_csv(data, f'smart_lamp_sensor_data_indonesia_{category}.csv')
        print(f"Generated {len(data)} sensor readings for {category} power category")
        print("Sample reading:")
        print(json.dumps(data[0], indent=2))