"""
AI Plant Care Engine
Handles growth stage calculation, AI model inference, care schedule generation,
and botanical intelligence synthesis in English.
"""

import os
import datetime
import pandas as pd
import joblib

from plants_data import PLANTS_DB

MODEL_FILE = os.path.join(os.path.dirname(__file__), "plant_care_model.joblib")

class PlantCareAI:
    def __init__(self):
        self.model_bundle = None
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model_bundle = joblib.load(MODEL_FILE)
            except Exception as e:
                print(f"Warning: Could not load ML model: {e}")
                self.model_bundle = None

    def get_plant_list(self):
        return list(PLANTS_DB.keys())

    def get_plant_info(self, plant_name):
        return PLANTS_DB.get(plant_name, None)

    def determine_growth_stage(self, plant_name, age_days):
        info = self.get_plant_info(plant_name)
        if not info:
            return "General Growth Stage", (0, 365), 50
        
        stages = info.get("growth_stages", {})
        current_stage_name = "Mature"
        stage_range = (120, 999)
        progress_pct = 100
        
        for stage_name, (start_day, end_day) in stages.items():
            if start_day <= age_days <= end_day:
                current_stage_name = stage_name
                stage_range = (start_day, end_day)
                span = max(1, end_day - start_day)
                progress_pct = min(100, int(((age_days - start_day) / span) * 100))
                break
        
        return current_stage_name, stage_range, progress_pct

    def auto_detect_season(self, month=None):
        if month is None:
            month = datetime.date.today().month
        if month in [3, 4, 5]:
            return "Summer", 0, 34.0, 50.0
        elif month in [6, 7, 8, 9]:
            return "Monsoon / Rainy Season", 1, 27.0, 85.0
        elif month in [10, 11]:
            return "Post-Monsoon / Autumn", 2, 29.0, 68.0
        else: # 12, 1, 2
            return "Winter", 3, 23.0, 58.0

    def predict_care(
        self,
        plant_name,
        planting_date,
        target_date=None,
        season_code=0,
        temp_c=32.0,
        humidity_pct=60.0,
        is_indoor=False,
        pot_size_code=1, # 0: Small, 1: Medium, 2: Large
        sunlight_hours=5.0
    ):
        if target_date is None:
            target_date = datetime.date.today()
            
        if isinstance(planting_date, datetime.datetime):
            planting_date = planting_date.date()
        if isinstance(target_date, datetime.datetime):
            target_date = target_date.date()
            
        age_days = max(1, (target_date - planting_date).days)
        plant_info = self.get_plant_info(plant_name)
        base_water = plant_info["base_water_ml"] if plant_info else 300
        
        stage_name, stage_range, stage_progress = self.determine_growth_stage(plant_name, age_days)
        
        # Plant Index
        plant_list = self.get_plant_list()
        plant_idx = plant_list.index(plant_name) if plant_name in plant_list else 0
        
        # ML Inference
        if self.model_bundle:
            feature_row = pd.DataFrame([{
                "plant_idx": plant_idx,
                "base_water": base_water,
                "age_days": age_days,
                "season_code": season_code,
                "temp_c": temp_c,
                "humidity_pct": humidity_pct,
                "is_indoor": 1 if is_indoor else 0,
                "pot_size": pot_size_code,
                "sunlight_hours": sunlight_hours
            }])
            
            water_ml = float(self.model_bundle["model_water"].predict(feature_row)[0])
            interval_days = float(self.model_bundle["model_interval"].predict(feature_row)[0])
        else:
            # Fallback calculation
            season_mult = [1.35, 0.55, 1.0, 0.75][season_code]
            pot_mult = [0.65, 1.0, 1.45][pot_size_code]
            indoor_mult = 0.7 if is_indoor else 1.0
            water_ml = base_water * season_mult * pot_mult * indoor_mult * (temp_c / 30.0)
            interval_days = 1.0 if season_code == 0 else 3.0

        water_ml = max(40, round(water_ml, -1))
        interval_days = max(1, round(interval_days))
        
        # Generate 7-day schedule
        schedule = []
        for i in range(7):
            day_date = target_date + datetime.timedelta(days=i)
            is_watering_day = (i % interval_days == 0)
            
            daily_water = water_ml if is_watering_day else 0
            schedule.append({
                "date": day_date.strftime("%Y-%m-%d"),
                "day_name": day_date.strftime("%A"),
                "water_ml": daily_water,
                "action": f"Water Thoroughly ({int(water_ml)} ml)" if is_watering_day else "Check Soil Moisture (No water needed)",
                "status": "Watering Day" if is_watering_day else "Resting Day"
            })
            
        # Season care recommendations
        if season_code == 0:
            season_name = "Summer"
            season_care_text = plant_info.get("summer_care", "") if plant_info else "Keep soil evenly moist. Water in the early morning to prevent high midday evaporation."
            pest_warning = "Watch for foliage sun-scorch, red spider mites, and severe dry soil crusting."
        elif season_code == 1:
            season_name = "Monsoon / Rainy Season"
            season_care_text = plant_info.get("monsoon_care", "") if plant_info else "Ensure pot drainage holes are free. Avoid stagnant water around root collars."
            pest_warning = "Monitor for fungal root rot, damping-off, snails, and waterlogged pot saucers."
        elif season_code == 2:
            season_name = "Post-Monsoon / Autumn"
            season_care_text = plant_info.get("tips", "") if plant_info else "Excellent growing climate. Good time for moderate fertilizing and branch trimming."
            pest_warning = "Inspect soft new shoots for aphids and caterpillar leaf-chewing."
        else:
            season_name = "Winter"
            season_care_text = plant_info.get("winter_care", "") if plant_info else "Reduce watering frequency. Keep in the sunniest accessible position."
            pest_warning = "Protect from cold nighttime drafts; guard against powdery mildew from damp standing moisture."

        return {
            "plant_name": plant_name,
            "scientific_name": plant_info.get("scientific_name", "") if plant_info else "",
            "category": plant_info.get("category", "") if plant_info else "",
            "age_days": age_days,
            "age_display": f"{age_days} Days ({age_days // 7} Weeks, {age_days % 7} Days)",
            "growth_stage": stage_name,
            "stage_range": stage_range,
            "stage_progress": stage_progress,
            "water_ml": int(water_ml),
            "interval_days": int(interval_days),
            "frequency_str": f"Every {interval_days} Day(s)" if interval_days > 1 else "Daily (Every Morning)",
            "season_name": season_name,
            "season_care": season_care_text,
            "pest_warning": pest_warning,
            "ideal_sunlight": plant_info.get("ideal_sunlight", "Moderate sunlight") if plant_info else "Moderate sunlight",
            "soil_info": plant_info.get("soil", "Well-draining fertile soil") if plant_info else "Well-draining fertile soil",
            "fertilizer_info": plant_info.get("fertilizer", "Balanced organic compost monthly") if plant_info else "Balanced organic compost monthly",
            "tips": plant_info.get("tips", "") if plant_info else "",
            "schedule": schedule
        }

care_engine = PlantCareAI()
