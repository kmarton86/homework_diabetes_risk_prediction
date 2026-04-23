from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

import config
from db import init_db
from ml_orchestrator import (
    init_models,
    get_dataset_summary,
    run_prediction,
    get_visualization_data
)


app = Flask(__name__)


# -----------------------
# Frontend  - Default

@app.route('/')
def index():
    return render_template(
        'index.html',
        current_threshold=config.CURRENT_THRESHOLD
    )

# -----------------------
# 1. Dataset Summary

@app.route('/api/dataset', methods=['GET'])
def dataset_summary():
    return jsonify(get_dataset_summary())

# -----------------------
# 2. Data Vizaulization

@app.route('/api/visualization', methods=['GET'])
def visualization_data():
    return jsonify(get_visualization_data())


# -----------------------
# 3. Patient input - Prediction

@app.route('/api/predict', methods=['POST'])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400
    
    """
        The incoming sex data (string) needs to be converted to numeric values. 
        These mapping datas are based on the stored dataset values. This is for trial purpose only - 
        the real conversion should be researched.

    """
    sex_map = {
        "female": -0.044642,
        "male": 0.050680 
    }

    try:
        patient = {
            "age": int(data.get("age")),
            "sex": float(sex_map.get(data.get("sex"))),
            "bmi": float(data.get("bmi")),
            "bp": float(data.get("bp")),
            "s1": float(data.get("s1")),
            "s2": float(data.get("s2")),
            "s3": float(data.get("s3")),
            "s4": float(data.get("s4")),
            "s5": float(data.get("s5")),
            "s6": float(data.get("s6"))
        }
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    print("Received patient:", patient)
    CURRENT_THRESHOLD = config.CURRENT_THRESHOLD
    print(f"current threshold: {CURRENT_THRESHOLD}")
    # convert dict -> DataFrame
    patient_df = pd.DataFrame([patient])
    prediction = run_prediction(patient_df, CURRENT_THRESHOLD)

    # test
    test_prediction = "Not Endangered"

    return jsonify({
        # "prediction": test_prediction,
        "prediction": prediction,
        "patient": patient
    })


# -----------------------
if __name__ == '__main__':
    # init database at startup
    DB_PATH = config.DB_PATH
    try:
        init_db(DB_PATH)
    except Exception as e:
        print(f"Error initializing database: {e}")

    # init models
    try:
        init_models()
    except Exception as e:
        print(f"Error initializing models: {e}")

    # app.run(debug=True)
    # for docker
    app.run(host="0.0.0.0", debug=True)