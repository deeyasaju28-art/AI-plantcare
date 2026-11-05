# 🌿 AI Plant Care System

An intelligent, machine-learning-powered plant care and irrigation management micro-project. The system calculates exact water requirements (in milliliters), watering intervals (frequency), plant growth stages, and season-specific care prescriptions based on botanical characteristics, planting date, and climatic factors.

---

## 📌 Key Features

1. **🌱 Growth Stage & Age Tracking:**
   - Computes plant age in days from the exact planting date and time.
   - Automatically tracks developmental phases: **Seedling**, **Vegetative Growth**, **Flowering / Budding**, and **Mature**, visualized via an interactive progress bar.

2. **💧 AI-Powered Irrigation Recommendation:**
   - Employs an ensemble **Random Forest Regressor** trained on agricultural evapotranspiration factors ($ET_0 \times K_c$).
   - Predicts the exact water volume needed (ml and standard glass equivalents) as well as the optimal watering interval (e.g., daily, every 2 days, weekly).

3. **📅 7-Day Smart Watering Schedule:**
   - Generates an interactive bar chart and calendar table detailing day-by-day watering instructions for the upcoming week.

4. **🍂 Season-Tailored Horticultural Care:**
   - Delivers customized seasonal advice for **Summer**, **Monsoon / Rainy Season**, **Autumn**, and **Winter**.
   - Prescribes soil medium mixes and organic fertilizer regimens (vermicompost, bone meal, seaweed extract, etc.).

5. **🩺 Plant Doctor & Symptom Diagnostic Assistant:**
   - Interactive diagnosis of common foliage problems (yellow leaves, crispy tips, wilting, powdery mildew, curled foliage).
   - Provides root-cause analysis and tested organic spray recipes (cold-pressed neem oil, baking soda fungicide, etc.).

6. **📚 Botanical Encyclopedia:**
   - Comprehensive profile cards for popular garden and indoor plants including scientific names, lighting requirements, and expert horticultural tips.

---

## 🛠️ Technology Stack

- **Core Programming Language:** Python 3
- **Web Application Framework:** Streamlit
- **Machine Learning:** Scikit-Learn (`RandomForestRegressor`)
- **Data Manipulation:** Pandas, NumPy
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Model Serialization:** Joblib

---

## 🚀 How to Run the Project

### Option A: One-Click Startup (Windows)
Simply double-click the `run_project.bat` file in the project folder.

### Option B: Terminal / Command Prompt
1. Open PowerShell or Command Prompt in the project directory:
   ```bash
   cd "c:\Users\deeya\OneDrive\Desktop\AI"
   ```
2. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   python -m streamlit run app.py
   ```
4. Open the application in your browser at `http://localhost:8502` or `http://localhost:8501`.

---

## 📂 Project Directory Structure

```text
c:\Users\deeya\OneDrive\Desktop\AI\
│
├── app.py                  # Main Streamlit web application and interactive UI
├── ai_engine.py            # AI inference engine, stage calculator, and scheduling logic
├── plants_data.py          # Botanical dataset and plant symptom diagnostic knowledge base
├── train_model.py          # Random Forest ML model training pipeline
├── plant_care_model.joblib # Serialized pre-trained machine learning model bundle
├── requirements.txt        # Python package dependencies
├── run_project.bat         # One-click Windows launch script
└── README.md               # Complete project documentation and viva guide
```

---

## 🎓 Academic Viva Questions & Answers

**Q1: What Machine Learning algorithm is used in this project?**
> **Answer:** An ensemble supervised learning algorithm — **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`). It uses an ensemble of decision trees to predict continuous values (water volume in ml and watering interval in days) with high generalization accuracy ($R^2 = 97.6\%$).

**Q2: What input features determine the watering prediction?**
> **Answer:**
> 1. Plant species baseline water requirement ($K_c$ crop factor)
> 2. Plant age and physiological growth stage (Seedling, Vegetative, Flowering, Mature)
> 3. Active climatic season (Summer, Monsoon, Autumn, Winter)
> 4. Microclimate parameters: Ambient temperature (°C) and relative humidity (%)
> 5. Daily sunlight duration (Hours/day)
> 6. Container size (Small, Medium, Large / In-ground)
> 7. Placement environment (Indoor vs. Outdoor)

**Q3: How does the planting date affect the calculation?**
> **Answer:** The planting timestamp is compared against the target date using Python's `datetime` module to calculate the plant's exact age in days. Plant species have distinct developmental stages: younger seedlings require lower water volume but higher soil moisture consistency, whereas flowering and fruiting stages have peak evapotranspiration demands.
