"""
AI Plant Care - Model Training Script
Generates botanical evapotranspiration synthetic dataset and trains
Random Forest Regressors for:
1. Water Quantity (ml per session)
2. Watering Interval (days between waterings)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plants_data import PLANTS_DB

def generate_botanical_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    plant_names = list(PLANTS_DB.keys())
    
    data = []
    
    for _ in range(n_samples):
        plant_idx = np.random.randint(0, len(plant_names))
        plant_name = plant_names[plant_idx]
        base_water = PLANTS_DB[plant_name]["base_water_ml"]
        
        # Plant Age in Days (1 to 365)
        age_days = np.random.randint(3, 365)
        
        # Season: 0=Summer, 1=Monsoon, 2=Spring/Autumn, 3=Winter
        season_code = np.random.choice([0, 1, 2, 3], p=[0.35, 0.25, 0.20, 0.20])
        
        # Temperature and Humidity according to season
        if season_code == 0:  # Summer
            temp = np.random.uniform(28.0, 42.0)
            humidity = np.random.uniform(30.0, 65.0)
            season_mult = 1.35
        elif season_code == 1:  # Monsoon
            temp = np.random.uniform(22.0, 31.0)
            humidity = np.random.uniform(75.0, 98.0)
            season_mult = 0.55
        elif season_code == 2:  # Spring / Moderate
            temp = np.random.uniform(23.0, 33.0)
            humidity = np.random.uniform(50.0, 75.0)
            season_mult = 1.0
        else:  # Winter
            temp = np.random.uniform(14.0, 26.0)
            humidity = np.random.uniform(40.0, 70.0)
            season_mult = 0.75

        # Indoor (1) vs Outdoor (0)
        is_indoor = np.random.choice([0, 1], p=[0.45, 0.55])
        
        # Pot size: 0: Small (<6 in), 1: Medium (6-12 in), 2: Large/Garden (>12 in)
        pot_size = np.random.choice([0, 1, 2], p=[0.3, 0.45, 0.25])
        pot_multipliers = [0.65, 1.0, 1.45]
        pot_mult = pot_multipliers[pot_size]
        
        # Growth Stage Factor
        if age_days < 20:
            stage_mult = 0.55  # Seedling needs smaller quantity, careful moisture
        elif age_days < 60:
            stage_mult = 0.90  # Vegetative growing
        elif age_days < 120:
            stage_mult = 1.20  # Peak bloom / active fruit
        else:
            stage_mult = 1.05  # Mature plant
            
        # Sunlight exposure hours
        sunlight_hours = np.random.uniform(1.0, 9.0)
        sun_mult = 0.8 + (sunlight_hours / 10.0) * 0.4
        
        # Evaporation index based on temp and humidity
        evap_index = (temp / 30.0) * (1.0 - (humidity / 150.0))
        
        # Calculate Target 1: Water amount in ml
        indoor_factor = 0.70 if is_indoor else 1.0
        calculated_water = base_water * stage_mult * season_mult * pot_mult * indoor_factor * evap_index
        # Add slight natural variation (Gaussian noise)
        calculated_water = calculated_water * np.random.uniform(0.95, 1.05)
        calculated_water = max(50.0, round(calculated_water, -1))  # round to nearest 10 ml
        
        # Calculate Target 2: Watering interval in days
        # E.g., High heat/outdoor = 1 day; Rain/winter/succulent = 3 to 14 days
        if "Snake" in plant_name or "Aloe" in plant_name:
            # Succulents
            if season_code == 0:
                interval_days = 7.0 if is_indoor else 4.0
            elif season_code == 1:
                interval_days = 14.0 if is_indoor else 10.0
            else:
                interval_days = 12.0 if is_indoor else 8.0
        else:
            # General foliage / herbs / flowering
            base_interval = 1.0
            if season_code == 1:  # Monsoon
                base_interval += 2.0
            elif season_code == 3:  # Winter
                base_interval += 1.5
            elif season_code == 0:  # Summer
                base_interval = 1.0
                
            if is_indoor:
                base_interval += 1.0
            if pot_size == 2:  # Large retains moisture longer
                base_interval += 0.5
            if humidity > 80:
                base_interval += 1.0
            if temp > 35:
                base_interval = max(1.0, base_interval - 1.0)
                
            interval_days = round(base_interval * np.random.uniform(0.9, 1.1))
            interval_days = max(1.0, min(14.0, interval_days))
            
        data.append({
            "plant_idx": plant_idx,
            "base_water": base_water,
            "age_days": age_days,
            "season_code": season_code,
            "temp_c": round(temp, 1),
            "humidity_pct": round(humidity, 1),
            "is_indoor": is_indoor,
            "pot_size": pot_size,
            "sunlight_hours": round(sunlight_hours, 1),
            "water_ml": calculated_water,
            "interval_days": interval_days
        })
        
    return pd.DataFrame(data)

def train_and_save_model():
    print("🌱 Generating botanical dataset...")
    df = generate_botanical_dataset(n_samples=6000)
    
    feature_cols = [
        "plant_idx", "base_water", "age_days", "season_code",
        "temp_c", "humidity_pct", "is_indoor", "pot_size", "sunlight_hours"
    ]
    
    X = df[feature_cols]
    y_water = df["water_ml"]
    y_interval = df["interval_days"]
    
    X_train, X_test, y_w_train, y_w_test, y_i_train, y_i_test = train_test_split(
        X, y_water, y_interval, test_size=0.2, random_state=42
    )
    
    print("🧠 Training Random Forest Regressor for Water Amount (ml)...")
    model_water = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42)
    model_water.fit(X_train, y_w_train)
    
    print("🧠 Training Random Forest Regressor for Watering Interval (Days)...")
    model_interval = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model_interval.fit(X_train, y_i_train)
    
    # Evaluation
    pred_w = model_water.predict(X_test)
    pred_i = model_interval.predict(X_test)
    
    r2_w = r2_score(y_w_test, pred_w)
    mae_w = mean_absolute_error(y_w_test, pred_w)
    r2_i = r2_score(y_i_test, pred_i)
    mae_i = mean_absolute_error(y_i_test, pred_i)
    
    print(f"✅ Water Model R² Score: {r2_w:.4f} | MAE: {mae_w:.2f} ml")
    print(f"✅ Interval Model R² Score: {r2_i:.4f} | MAE: {mae_i:.2f} days")
    
    model_bundle = {
        "model_water": model_water,
        "model_interval": model_interval,
        "feature_cols": feature_cols,
        "metrics": {
            "r2_water": r2_w,
            "mae_water": mae_w,
            "r2_interval": r2_i,
            "mae_interval": mae_i
        }
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "plant_care_model.joblib")
    joblib.dump(model_bundle, output_path)
    print(f"🎉 Model bundle saved successfully to {output_path}")

if __name__ == "__main__":
    train_and_save_model()
