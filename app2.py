import os
import joblib
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Paths to your model files
decision_tree_model_path = 'C:/Users/USER/Desktop/credit_card_fraud_prediction/best_decision_tree_model.joblib'
logistic_regression_model_path = 'C:/Users/USER/Desktop/credit_card_fraud_prediction/best_logistic_regression_model.joblib'
ann_model_path = 'C:/Users/USER/Desktop/credit_card_fraud_prediction/best_ann_model.keras'

# Load the models
try:
    decision_tree_model = joblib.load(decision_tree_model_path)
    print("Decision Tree model loaded successfully.")
except Exception as e:
    print(f"Error loading Decision Tree model: {e}")

try:
    logistic_regression_model = joblib.load(logistic_regression_model_path)
    print("Logistic Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading Logistic Regression model: {e}")

try:
    ann_model = tf.keras.models.load_model(ann_model_path)
    print("ANN model loaded successfully.")
except Exception as e:
    print(f"Error loading ANN model: {e}")

@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML form for prediction

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form  # Since we are submitting a form, we use form data (not JSON)

    # Extract features from the form (assuming the form inputs are named "feature1", "feature2", ..., "feature23")
    features = []
    for i in range(1, 24):  # Looping for feature1 to feature23
        try:
            feature_value = float(data.get(f'feature{i}'))
            features.append(feature_value)
        except ValueError:
            return jsonify({"error": f"Invalid input for feature{i}. Please enter numeric values."})

    features = np.array(features).reshape(1, -1)  # Reshape for prediction (1 sample, 23 features)

    # Make predictions using each model
    try:
        dt_prediction = decision_tree_model.predict(features)
        dt_prediction = int(dt_prediction[0])  # Convert to int for a clear 0 or 1 result
    except Exception as e:
        dt_prediction = f"Error in Decision Tree Prediction: {e}"

    try:
        lr_prediction = logistic_regression_model.predict(features)
        lr_prediction = int(lr_prediction[0])  # Convert to int for a clear 0 or 1 result
    except Exception as e:
        lr_prediction = f"Error in Logistic Regression Prediction: {e}"

    try:
        ann_prediction = ann_model.predict(features)
        # Apply threshold: if prediction >= 0.5, return 1; else return 0
        ann_prediction = 1 if ann_prediction[0][0] >= 0.5 else 0
    except Exception as e:
        ann_prediction = f"Error in ANN Prediction: {e}"

    # Return the predictions as a JSON response
    return jsonify({
        'decision_tree_prediction': dt_prediction,
        'logistic_regression_prediction': lr_prediction,
        'ann_prediction': ann_prediction
    })

if __name__ == "__main__":
    app.run(debug=True)