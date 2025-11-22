from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ---- Configuration ----
MODEL_BUNDLE_PATH = "house_price_predictor_model.pkl"  # must exist in same dir
MIN_AREA = 200  # same rule as safe_predict_v2

app = Flask(__name__, static_folder='.', static_url_path='')

# ---- Load model bundle ----
if not os.path.exists(MODEL_BUNDLE_PATH):
    raise FileNotFoundError(f"Model bundle not found: {MODEL_BUNDLE_PATH}. Place it in the same folder as app.py")

import sklearn
sklearn.set_config(assume_finite=True)
bundle = joblib.load(MODEL_BUNDLE_PATH)
model = bundle.get("model")
ppsf_q05 = bundle.get("ppsf_q05")
ppsf_q95 = bundle.get("ppsf_q95")
loc_medians_enc = bundle.get("loc_medians_enc", {})
global_base_ppsf_median_enc = bundle.get("global_base_ppsf_median_enc")

# Safety: ensure numeric values exist
def _impute_base_ppsf_row(row):
    b = row.get('base_ppsf', np.nan)
    if pd.notna(b) and b > 0:
        return b
    loc = row.get('location_type', None)
    if pd.notna(loc) and loc in loc_medians_enc:
        return loc_medians_enc[loc]
    return global_base_ppsf_median_enc

@app.route('/')
def index():
    # Serve index.html
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify(error="Invalid JSON body"), 400

    # Basic validation
    try:
        area = float(data.get('area', 0))
        bedrooms = int(data.get('bedrooms', 0))
        bathrooms = int(data.get('bathrooms', 0))
    except Exception:
        return jsonify(error="Invalid numeric fields"), 400

    if area < MIN_AREA:
        return jsonify(error=f"Invalid input: area must be at least {MIN_AREA} sq ft."), 400
    if bedrooms <= 0 or bathrooms <= 0:
        return jsonify(error="Bedrooms and bathrooms must be positive integers."), 400

    # Build DataFrame row for model (columns must match training X)
    # Column order must match your training data. This code assumes the order used earlier.
    row = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': int(data.get('stories', 1)),
        'mainroad': int(data.get('mainroad', 0)),
        'guestroom': int(data.get('guestroom', 0)),
        'basement': int(data.get('basement', 0)),
        'hotwaterheating': int(data.get('hotwaterheating', 0)),
        'airconditioning': int(data.get('airconditioning', 0)),
        'parking': int(data.get('parking', 0)),
        'prefarea': int(data.get('prefarea', 0)),
        'furnishingstatus': int(data.get('furnishingstatus', 0)),
        'location_type': int(data.get('location_type', 0)),
        'base_ppsf': float(data.get('base_ppsf', 0))
    }

    # Impute base_ppsf
    row['base_ppsf'] = _impute_base_ppsf_row(row)

    # Prepare DataFrame with single row
    x = pd.DataFrame([row])

    # Predict
    try:
        raw_pred = float(model.predict(x)[0])
    except Exception as e:
        return jsonify(error=f"Model prediction error: {str(e)}"), 500

    # Clamp prediction by ppsf quantiles (based on training)
    area_val = float(x.loc[0, 'area'])
    lower_bound = area_val * ppsf_q05
    upper_bound = area_val * ppsf_q95
    bounded_pred = float(np.clip(raw_pred, lower_bound, upper_bound))

    # Minimum floor (business rule from your code)
    bounded_pred = max(bounded_pred, 500000.0)

    # Respond with clean message
    message = f"Price: ₹{bounded_pred:,.0f} (≈ ₹{bounded_pred/100000:.2f} lakh)"
    return jsonify(
        message=message,
        raw_prediction=raw_pred,
        bounded_prediction=bounded_pred,
        clamp_lower=lower_bound,
        clamp_upper=upper_bound,
    ), 200

if __name__ == "__main__":
    # Run dev server on localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
