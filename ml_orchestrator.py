import pandas as pd
import json # for testing

from db import load_from_db
from ml.model import create_labels, train_model
from ml.prediction import predict_patient
from ml.analysis import dataset_summary, class_distribution

DB_PATH = "diabetes.db"
THRESHOLD_150 = 150
THRESHOLD_250 = 250

CURRENT_THRESHOLD = THRESHOLD_150

# -----------------------
# LOAD DATA
# -----------------------
"""
X = bemeneti jellemzők (features): age, bmi, bp, stb.
y = eredmény (target) -> drop oka, hogy ezt a ML-nek kell megtanulnia, nem adhatjuk át
"""
def get_X_y_from_dataset():
    df = load_from_db(DB_PATH)
    X = df.drop(columns=["id", "target"])
    y = df["target"]
    return X, y

# -----------------------
# TRAIN MODELS - for both scenarios
# -----------------------
MODELS = None

def train_all_models():
    # get X,y from db to train models
    X, y = get_X_y_from_dataset()

    # create_labels, train_model from ml.model
    y_150 = create_labels(y, 150)
    y_250 = create_labels(y, 250)

    model_150, scaler_150, metrics_150  = train_model(X, y_150)
    model_250, scaler_250, metrics_250 = train_model(X, y_250)

    return {
        150: {
            "model": model_150,
            "scaler": scaler_150,
            "metrics": metrics_150
        },
        250: {
            "model": model_250,
            "scaler": scaler_250,
            "metrics": metrics_250
        }
    }

# train all models - call it once from app.py
def init_models():
    global MODELS
    MODELS = train_all_models()

# -----------------------
# ML ANALYSIS
# -----------------------

# DATASET SUMMARY
def get_dataset_summary():
    X, y = get_X_y_from_dataset()
    return dataset_summary(X, y)

# CLASS DISTRIBUTION
def get_class_distribution(threshold):
    # only y is needed for class_distribution
    _, y = get_X_y_from_dataset()
    return class_distribution(y, threshold)


# -----------------------
# MODEL PERFORMANCE 
# -----------------------
def get_model_performance():
    global MODELS

    if MODELS is None:
        init_models()

    return {
        "150": MODELS[150]["metrics"],
        "250": MODELS[250]["metrics"]
    }

# -----------------------
# VISUALIZATION 
# -----------------------

def get_visualization_data():
    X, y = get_X_y_from_dataset()

    return {
        "dataset_summary": dataset_summary(X, y),
        "class_distribution_150": class_distribution(y, 150),
        "class_distribution_250": class_distribution(y, 250),
        "model_performance": get_model_performance()  # model külön jön
    }
# -----------------------
# PREDICTION
# -----------------------
def run_prediction(patient_df, threshold):
    global MODELS

    if MODELS is None:
        init_models()

    model = MODELS[threshold]["model"]
    scaler = MODELS[threshold]["scaler"]

    return predict_patient(patient_df, model, scaler)

# -----------------------
# TESTING
# -----------------------
if __name__ == "__main__":

    init_models()  

    print("\nDATASET SUMMARY:")
    print(get_dataset_summary())

    print("\nCLASS DISTRIBUTION (150):")
    print(get_class_distribution(THRESHOLD_150))

    print("\nCLASS DISTRIBUTION (250):")
    print(get_class_distribution(THRESHOLD_250))

    print("\nMODEL PERFORMANCE:")
    print(get_model_performance())

    new_patient = {
        "age": 50,
        "sex": 0,
        "bmi": 28,
        "bp": 120,
        "s1": 85,
        "s2": 180,
        "s3": 40,
        "s4": 5,
        "s5": 4.5,
        "s6": 90
    }

    patient_df = pd.DataFrame([new_patient])

    print(f"\nPREDICTION (threshold = {CURRENT_THRESHOLD}):")
    print(run_prediction(patient_df, CURRENT_THRESHOLD))

    # -----------------------
    # GET VISUALIZATION BUNDLE
    # -----------------------
    viz_data = get_visualization_data()

    # -----------------------
    # PRINT RAW
    # -----------------------
    print("\nRAW VISUALIZATION DATA:\n")
    print(viz_data)

    # -----------------------
    # PRETTY JSON (frontend-ready check)
    # -----------------------

    print("\nPRETTY JSON:\n")
    print(json.dumps(viz_data, indent=4))