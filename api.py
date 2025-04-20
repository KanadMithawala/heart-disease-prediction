from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load models and scaler
rf_model = joblib.load('rf_model.joblib')
gb_model = joblib.load('gb_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def serve_gui():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data['male'], data['age'], data['currentSmoker'], data['cigsPerDay'],
            data['BPMeds'], data['prevalentStroke'], data['prevalentHyp'], data['diabetes'],
            data['totChol'], data['sysBP'], data['diaBP'], data['BMI'],
            data['heartRate'], data['glucose']
        ]
        features_scaled = scaler.transform([features])
        rf_prob = rf_model.predict_proba(features_scaled)[0][1]
        gb_prob = gb_model.predict_proba(features_scaled)[0][1]
        ensemble_prob = (rf_prob + gb_prob) / 2
        prediction = 'High Risk' if ensemble_prob > 0.5 else 'Low Risk'
        return jsonify({'prediction': prediction, 'probability': ensemble_prob})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)