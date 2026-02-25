# Mileage Predictor AI

## Setup
1. Install dependencies:
   ```
   pip install flask scikit-learn joblib pandas numpy
   ```
2. Run the server:
   ```
   python app.py
   ```
3. Open your browser at: http://localhost:5050

## Files
- `app.py` — Flask backend (loads model, serves API)
- `static/index.html` — Frontend UI
- `mileage_model.pkl` — Trained Random Forest model
- `city_traffic_scores.csv` — City traffic congestion scores
