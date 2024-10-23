import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_smart_lamp_data_indonesia(start_date, num_days):
    # Indonesia typically has 12 hours of daylight year-round
    sunrise_time = 6  # 6 AM
    sunset_time = 18  # 6 PM

    date_range = pd.date_range(start=start_date, periods=num_days*32, freq='45T')
    data = []

    for timestamp in date_range:
        day_of_week = timestamp.dayofweek
        hour = timestamp.hour
        minute = timestamp.minute

        # Simulate ambient light levels (adjusted for tropical climate)
        if sunrise_time <= hour < sunset_time:
            if 11 <= hour < 13:  # Midday brightness
                ambient_light = np.random.normal(800, 100)
            else:
                ambient_light = np.random.normal(600, 150)
        else:
            ambient_light = np.random.normal(30, 20)
        ambient_light = max(0, min(1000, ambient_light))

        # Simulate room occupancy (adjusted for typical Indonesian household patterns)
        if 5 <= hour < 22:  # Longer active hours
            room_occupied = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            room_occupied = np.random.choice([0, 1], p=[0.9, 0.1])

        # Current lamp settings
        if room_occupied:
            current_brightness = max(0, min(100, np.random.normal(60, 20)))
            if sunrise_time <= hour < sunset_time:
                current_color_temp = np.random.normal(4500, 500)  # Cooler during the day
            else:
                current_color_temp = np.random.normal(2700, 300)  # Warmer at night
        else:
            current_brightness = 0
            current_color_temp = 0

        current_color_temp = max(2000, min(6500, current_color_temp))

        # Energy consumption (consider higher efficiency due to tropical climate awareness)
        energy_consumption = current_brightness * 0.008 * np.random.normal(1, 0.1)

        # User override (slightly higher chance due to varied preferences)
        user_override = np.random.choice([0, 1], p=[0.93, 0.07])

        # Optimal settings (adjusted for tropical preferences)
        if room_occupied:
            optimal_brightness = max(0, min(100, np.random.normal(65, 15) - (ambient_light / 25)))
            if sunrise_time <= hour < sunset_time:
                optimal_color_temp = max(2000, min(6500, np.random.normal(5000, 300) - (hour - 12) * 80))
            else:
                optimal_color_temp = max(2000, min(6500, np.random.normal(2500, 200) + (hour - 18) * 40))
        else:
            optimal_brightness = 0
            optimal_color_temp = 0

        data.append([
            timestamp, day_of_week, hour, ambient_light, room_occupied,
            current_brightness, current_color_temp, energy_consumption,
            user_override, optimal_brightness, optimal_color_temp
        ])

    df = pd.DataFrame(data, columns=[
        'Timestamp', 'Day_of_Week', 'Hour_of_Day', 'Ambient_Light_Level',
        'Room_Occupied', 'Current_Lamp_Brightness', 'Current_Lamp_Color_Temp',
        'Energy_Consumption', 'User_Override', 'Optimal_Brightness', 'Optimal_Color_Temp'
    ])

    return df

# Generate data for one month
start_date = '2024-10-01'
num_days = 31
df = generate_smart_lamp_data_indonesia(start_date, num_days)

# Save to CSV
df.to_csv('smart_lamp_data_indonesia.csv', index=False)
print("Data saved to smart_lamp_data_indonesia.csv")