from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_diabetes

app = Flask(__name__)


# =========================
# FRONTEND ENTRY
# =========================
@app.route('/')
def index():
    return render_template('index.html')


# =========================
# 1. DATASET SUMMARY API
# =========================
@app.route('/api/dataset', methods=['GET'])
def dataset_summary():
    data = load_diabetes()

    summary = {
        "samples": len(data.data),
        "features": len(data.feature_names),
        "feature_names": data.feature_names
    }

    return jsonify(summary)


# =========================
# 2. VISUALIZATION DATA API
# =========================
@app.route('/api/visualization', methods=['GET'])
def visualization_data():
    data = load_diabetes()

    # skeleton: csak nyers adatot adunk vissza
    # frontend (Chart.js) fogja rajzolni

    response = {
        "bmi": data.data[:, 2].tolist(),
        "bp": data.data[:, 3].tolist(),
        "target": data.target.tolist()
    }

    return jsonify(response)


# =========================
# 3. PREDICTION API
# =========================
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    sex_map = {
        "female": 0,
        "male": 1
    }

    try:
        patient = {
            "age": int(data.get("age")),
            "sex": sex_map.get(data.get("sex")),
            "bmi": float(data.get("bmi")),
            "bp": int(data.get("bp")),
            "s1": int(data.get("s1")),
            "s2": int(data.get("s2")),
            "s3": int(data.get("s3")),
            "s4": int(data.get("s4")),
            "s5": float(data.get("s5")),
            "s6": int(data.get("s6"))
        }
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    print("Received patient:", patient)

    # skeleton prediction (ML később jön ide)
    fake_prediction = "Not Endangered"

    return jsonify({
        "prediction": fake_prediction,
        "patient": patient
    })


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)