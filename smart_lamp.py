
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta 

# Function to generate synthetic smart lamp data for given number of days
# def generate_smart_lamp_data(days=30):
#     # Set start date for data generation
#     start_date = datetime(2024, 10, 1)
#     # Create list of timestamps at 45-minute intervals
#     dates = [start_date + timedelta(minutes=45*i) for i in range(days*24*60//45)]
    
#     data = []  # Initialize empty list to store data
#     for date in dates:
#         # Generate ambient light level using sine wave pattern (simulating daylight)
#         # Multiply normal distribution with sine squared for realistic light patterns
#         ambient_light = np.random.normal(500, 200) * np.sin(np.pi * date.hour / 12)**2
#         ambient_light = max(0, min(1000, ambient_light))  # Limit light level between 0-1000
        
#         # Generate room occupancy (70% chance of being occupied)
#         room_occupied = np.random.choice([0, 1], p=[0.3, 0.7])
        
#         if room_occupied:
#             # Generate lamp settings for occupied room
#             brightness = np.random.normal(50, 20)  # Normal distribution around 50%
#             brightness = max(0, min(100, brightness))  # Limit brightness between 0-100%
#             color_temp = np.random.normal(4000, 1000)  # Normal distribution around 4000K
#             color_temp = max(2000, min(6500, color_temp))  # Limit color temp between 2000-6500K
            
#             # Calculate energy consumption based on multiple factors
#             base_energy = brightness * 0.01  # Base energy proportional to brightness
#             time_factor = 1 - 0.3 * np.sin(np.pi * date.hour / 12)  # Time-based efficiency
#             ambient_factor = 1 - 0.5 * (ambient_light / 1000)  # Ambient light efficiency
            
#             # Calculate final energy consumption
#             energy_consumption = base_energy * time_factor * ambient_factor
#             energy_consumption = max(0, energy_consumption)  # Ensure non-negative
#         else:
#             # Set default values for unoccupied room
#             brightness = 0
#             color_temp = 2000
#             energy_consumption = 0
        
#         # Append data point to list
#         data.append({
#             'Timestamp': date,
#             'Day_of_Week': date.weekday(),
#             'Hour_of_Day': date.hour,
#             'Ambient_Light_Level': ambient_light,
#             'Room_Occupied': room_occupied,
#             'Current_Lamp_Brightness': brightness,
#             'Current_Lamp_Color_Temp': color_temp,
#             'Energy_Consumption': energy_consumption
#         })
    
#     # Convert list of dictionaries to DataFrame
#     return pd.DataFrame(data)

# Generate dataset and save to CSV
# df = generate_smart_lamp_data(days=30)
# csv_filename = "smart_lamp_data.csv"
# df.to_csv(csv_filename, index=False)
# print(f"Data saved to {csv_filename}")



# Dataset
# We prepared a dataset simulating 30 days of smart lamp usage. The data includes natural light levels throughout the day, when rooms are occupied, and how much energy the lamp uses at different settings. 

# Load CSV file
df = pd.read_csv("smart_lamp_data.csv")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Define feature columns and target variable
features = ['Ambient_Light_Level', 'Room_Occupied', 'Hour_of_Day', 'Day_of_Week', 
           'Current_Lamp_Brightness', 'Current_Lamp_Color_Temp']
target = 'Energy_Consumption'

# Prepare features (X) and target (y)
X = df[features]
y = df[target]

# Split data into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_val_scaled = scaler.transform(X_val)  # Transform validation data using training scaler


#Preprocessing
# The preprocessing step transforms raw lamp data into a format suitable for machine learning. It starts by loading historical data of lamp usage, converting time information into usable features, and scaling all inputs to similar ranges (typically between -1 and 1). This scaling is crucial because it prevents features with larger numbers (like color temperature at 6500K) from overwhelming features with smaller numbers (like brightness at 100%). The end result is a clean, normalized dataset ready for training.

# Create neural network model
model = Sequential([
    # Input layer with 64 neurons
    Dense(64, activation='relu', input_shape=(len(features),)),
    Dropout(0.2),  # 20% dropout for regularization
    # Hidden layer with 32 neurons
    Dense(32, activation='relu'),
    Dropout(0.2),  # 20% dropout
    # Hidden layer with 16 neurons
    Dense(16, activation='relu'),
    # Output layer with 1 neuron for regression
    Dense(1)
])

# Compile model with Adam optimizer and MSE loss
model.compile(optimizer=Adam(learning_rate=0.001), 
             loss='mean_squared_error', 
             metrics=['mae'])

# Starting with 64 neurons allows the network to capture complex patterns, while each subsequent layer reduces this complexity until reaching a single prediction. ReLU activation functions help the network learn non-linear patterns, while dropout layers prevent overfitting by randomly disabling neurons during training. The architecture is then compiled using the Adam (Adaptive Moment Estimation) optimizer, which adaptively adjusts learning rates for each network parameter.

# Train the model
history = model.fit(
    X_train_scaled, y_train,  # Use training data
    epochs=1000,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    verbose=1
)

######################################### IMPROVEMENT #######################################
improved_model = Sequential([
    Dense(128, activation='relu', input_shape=(len(features),)),  # Larger first layer
    BatchNormalization(),  # Add batch normalization
    Dense(64, activation='relu'),
    Dropout(0.3),  # Adjusted dropout rate
    Dense(32, activation='relu'),
    Dense(1)
])

# 2. Adjust compilation parameters
improved_model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
    loss='mean_squared_error',
    metrics=['mae']
)

# 3. Train with modified parameters
history_improved = improved_model.fit(
    X_train_scaled, y_train,
    epochs=1500,  # More epochs
    batch_size=64,  # Larger batch size
    validation_data=(X_val_scaled, y_val),
    verbose=1
)

original_val_loss = model.evaluate(X_val_scaled, y_val)[0]
improved_val_loss = improved_model.evaluate(X_val_scaled, y_val)[0]
print(f"Original Model Validation Loss: {original_val_loss:.4f}")
print(f"Improved Model Validation Loss: {improved_val_loss:.4f}")
# Visualization of training history
plt.figure(figsize=(12, 4))
# Plot loss curves
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MAE curves
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Function to predict energy consumption for given conditions
def predict_energy_consumption(ambient_light, room_occupied, hour, day, brightness, color_temp):
    # Prepare input data
    new_data = np.array([[ambient_light, room_occupied, hour, day, brightness, color_temp]])
    # Scale input using same scaler as training data
    new_data_scaled = scaler.transform(new_data)
    # Uses the learned patterns from model.fit to make prediction
    prediction = model.predict(new_data_scaled)
    return prediction[0][0]




# Test predict_energy_consumption with example values
ambient_light = 300
room_occupied = 1  # 1 means room is occupied
hour = 20
day = 3
brightness = 50    # 50% brightness
color_temp = 4000  # 4000K color temperature

# Get prediction
predicted_energy = predict_energy_consumption(
    ambient_light, room_occupied, hour, day, brightness, color_temp
)

print(f"\nPrediction for:")
print(f"Ambient Light: {ambient_light}")
print(f"Room Occupied: {room_occupied}")
print(f"Hour: {hour}")
print(f"Day: {day}")
print(f"Brightness: {brightness}%")
print(f"Color Temperature: {color_temp}K")
print(f"Predicted Energy Consumption: {predicted_energy:.4f}") #prediction value


######################################################################################################################

#Function to find optimal lamp settings
def optimize_lamp_settings(ambient_light, hour, day, min_brightness=5):
    best_energy = float('inf')
    best_settings = None
    # Try different combinations of brightness and color temperature
    for brightness in range(min_brightness, 101, 5):  # 5% steps
        for color_temp in range(2000, 6501, 500):  # 500K steps
            # Calculate energy consumption for each combination
            energy = predict_energy_consumption(ambient_light, 1, hour, day, brightness, color_temp)
            # Update if better than current best
            if energy < best_energy:
                best_energy = energy
                best_settings = (brightness, color_temp)
    return best_settings, best_energy

# Test optimization with example scenario
ambient_light = 300
hour = 20
day = 3
min_brightness = 5

# Get optimal settings and print results
optimal_settings, optimal_energy = optimize_lamp_settings(ambient_light, hour, day, min_brightness)
print(f"Optimal settings for ambient light {ambient_light}, hour {hour}, day {day}:")
print(f"Brightness: {optimal_settings[0]}%, Color Temperature: {optimal_settings[1]}K")
print(f"Estimated Energy Consumption: {optimal_energy:.4f}")

# Analyze and print feature importance
feature_importance = model.layers[0].get_weights()[0]
for feature, importance in zip(features, np.mean(np.abs(feature_importance), axis=1)):
    print(f"{feature}: {importance:.4f}")


######################################################################################################################

# Test model with different scenarios
scenarios = [
    (100, 8, 1),  # Low light, morning, weekday
    (500, 12, 3), # Medium light, noon, midweek
    (800, 15, 5), # Bright light, afternoon, Friday
    (200, 20, 6)  # Evening, weekend
]

# Print recommendations for each scenario
print("\nRecommendations for different scenarios:")
for ambient_light, hour, day in scenarios:
    optimal_settings, optimal_energy = optimize_lamp_settings(ambient_light, hour, day, min_brightness)
    print(f"\nScenario: Ambient Light: {ambient_light}, Hour: {hour}, Day: {day}")
    print(f"Recommended Brightness: {optimal_settings[0]}%")
    print(f"Recommended Color Temperature: {optimal_settings[1]}K")
    print(f"Estimated Energy Consumption: {optimal_energy:.4f}")