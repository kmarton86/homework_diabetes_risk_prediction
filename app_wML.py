from flask import Flask, request, jsonify, render_template
import pandas as pd

from ml.dataset import load_data
from ml.model import create_labels, train_model
from ml.prediction import predict_patient

app = Flask(__name__)

# -----------------------
# INIT MODEL (startup)
# -----------------------
X, y = load_data()

y_150 = create_labels(y, 150)
y_250 = create_labels(y, 250)

model_150, scaler_150 = train_model(X, y_150)
model_250, scaler_250 = train_model(X, y_250)

# -----------------------
# ROUTES
# -----------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    new_patient = {
        "age": float(data["age"]),
        "sex": float(data["sex"]),
        "bmi": float(data["bmi"]),
        "bp": float(data["bp"]),
        "s1": float(data["s1"]),
        "s2": float(data["s2"]),
        "s3": float(data["s3"]),
        "s4": float(data["s4"]),
        "s5": float(data["s5"]),
        "s6": float(data["s6"]),
    }

    patient_df = pd.DataFrame([new_patient])

    result_150 = predict_patient(patient_df, model_150, scaler_150)
    result_250 = predict_patient(patient_df, model_250, scaler_250)

    return jsonify({
        "150": result_150,
        "250": result_250
    })


if __name__ == "__main__":
    app.run(debug=True)