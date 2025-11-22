"""
Script to retrain the model with current scikit-learn version.
This creates a simple fallback model that can work with the Flask app.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Create a simple model bundle that matches the expected structure
# Since we don't have the original training data, we'll create a basic model
# that can at least run (though predictions won't be accurate without retraining on real data)

print("Creating a new model bundle compatible with current scikit-learn version...")

# Create a simple model (this should be retrained with actual data)
model = RandomForestRegressor(n_estimators=10, random_state=42)

# Create dummy training data with the expected features
n_samples = 100
X_dummy = np.random.rand(n_samples, 14)  # 14 features
y_dummy = np.random.rand(n_samples) * 5000000 + 1000000  # Random prices

# Fit the model
model.fit(X_dummy, y_dummy)

# Create the bundle with the same structure as expected
bundle = {
    "model": model,
    "ppsf_q05": 3000.0,  # Placeholder values
    "ppsf_q95": 15000.0,
    "loc_medians_enc": {0: 5000, 1: 7000, 2: 9000},  # Placeholder location medians
    "global_base_ppsf_median_enc": 6000.0
}

# Save the bundle
output_path = "house_price_predictor_model.pkl"
joblib.dump(bundle, output_path)
print(f"✓ Model bundle saved to {output_path}")
print("\nNOTE: This is a placeholder model created for compatibility.")
print("For accurate predictions, you should retrain with your actual housing dataset.")
