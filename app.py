import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import csv
import os

app = Flask(__name__, static_folder='static')

# Load model
obj = joblib.load('mileage_model.pkl')
model = obj['model']
features = obj['features']

# Fix SimpleImputer _fill_dtype bug (sklearn version mismatch)
def fix_imputer(imputer):
    if not hasattr(imputer, '_fill_dtype') and hasattr(imputer, 'statistics_'):
        imputer._fill_dtype = imputer.statistics_.dtype

ct = model.steps[0][1]
for name, pipe, cols in ct.transformers_:
    for step_name, step in pipe.steps:
        if hasattr(step, 'statistics_'):
            fix_imputer(step)

# Load city traffic scores
city_scores = {}
with open('city_traffic_scores.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        city_scores[row['CITY']] = int(row['TRAFFIC_SCORE'])

# Get model categories
cat_pipe = ct.transformers_[1][1]
ohe = cat_pipe.steps[-1][1]
cat_cols = ct.transformers_[1][2]
categories = {col: sorted(ohe.categories_[i].tolist()) for i, col in enumerate(cat_cols)}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/metadata')
def metadata():
    return jsonify({
        'brands': categories['Brand'],
        'models': categories['Model'],
        'fuel_types': categories['Fuel_type'],
        'transmissions': categories['Transmission'],
        'cities': sorted(city_scores.keys()),
        'city_scores': city_scores
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    city = data.get('city')
    traffic_score = city_scores.get(city, 50)
    
    sample = pd.DataFrame({
        'Total_km_driven': [float(data['total_km'])],
        'Brand': [data['brand']],
        'Model': [data['model']],
        'Engine_cc': [float(data['engine_cc'])],
        'Fuel_type': [data['fuel_type']],
        'City_traffic_score': [traffic_score],
        'Transmission': [data['transmission']],
        'Age_of_vehicle': [float(data['age'])],
        'Number_of_owners': [int(data['owners'])]
    })
    
    pred = model.predict(sample)[0]
    
    # Feature importances from the RF
    rf = model.steps[-1][1]
    importances = rf.feature_importances_.tolist()
    feature_names_out = model.steps[0][1].get_feature_names_out().tolist()
    
    # Group importances back to original features
    original_features = ['Total_km_driven', 'Engine_cc', 'City_traffic_score', 'Age_of_vehicle', 'Number_of_owners', 'Brand', 'Model', 'Fuel_type', 'Transmission']
    grouped = {}
    for fname in original_features:
        total = sum(imp for fn, imp in zip(feature_names_out, importances) 
                   if fname.lower() in fn.lower() or fn.endswith(fname))
        grouped[fname] = round(total, 4)
    
    return jsonify({
        'prediction': round(float(pred), 2),
        'range_low': round(float(pred) * 0.9, 2),
        'range_high': round(float(pred) * 1.1, 2),
        'traffic_score': traffic_score,
        'importances': grouped
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
