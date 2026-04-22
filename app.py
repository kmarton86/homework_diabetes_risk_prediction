from flask import Flask, render_template, request, jsonify
import pandas as pd

from ml_orchestrator import (
    init_models,
    get_dataset_summary,
    get_class_distribution,
    run_prediction,
    get_visualization_data
)
from db import init_db

app = Flask(__name__)

# -----------------------
#  App config
# -----------------------

# Database
DB_PATH = "diabetes.db"

# Model target thresholds
THRESHOLD_150 = 150
THRESHOLD_250 = 250

CURRENT_THRESHOLD = THRESHOLD_150


# -----------------------
# FRONTEND ENTRY
# -----------------------
@app.route('/')
def index():
    return render_template(
        'index.html',
        current_threshold=CURRENT_THRESHOLD
    )


# -----------------------
# 1. DATASET SUMMARY API
# -----------------------
@app.route('/api/dataset', methods=['GET'])
def dataset_summary():
    return jsonify(get_dataset_summary())


# -----------------------
# 2. VISUALIZATION DATA API
# -----------------------
@app.route('/api/visualization', methods=['GET'])
def visualization_data():
    return jsonify(get_visualization_data())


# -----------------------
# 3. PREDICTION API
# -----------------------
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
    # convert dict -> DataFrame
    patient_df = pd.DataFrame([patient])
    prediction = run_prediction(patient_df, THRESHOLD_150)

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
    try:
        init_db(DB_PATH)
    except Exception as e:
        print(f"Error initializing database: {e}")

    # init models
    try:
        init_models()
    except Exception as e:
        print(f"Error initializing models: {e}")

    app.run(debug=True)