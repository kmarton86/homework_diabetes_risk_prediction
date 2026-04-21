import pandas as pd

from db import load_from_db
from ml.model import create_labels, train_model
from ml.prediction import predict_patient
from ml.analysis import dataset_summary, class_distribution

DB_PATH = "diabetes.db"


# -----------------------
# LOAD DATA
# -----------------------
def get_dataset(DB_PATH):
    df = load_from_db(DB_PATH)
    return df

"""
X = bemeneti jellemzők (features): age, bmi, bp, stb.
y = eredmény (target) -> drop oka, hogy ezt a ML-nek kell megtanulnia, nem adhatjuk át
"""
def get_X_y():
    df = get_dataset()
    X = df.drop(columns=["id", "target"])
    y = df["target"]
    return X, y


# -----------------------
# TRAIN MODELS
# -----------------------
def train_models():
    X, y = get_X_y()

    y_150 = create_labels(y, 150)
    y_250 = create_labels(y, 250)

    model_150, scaler_150 = train_model(X, y_150)
    model_250, scaler_250 = train_model(X, y_250)

    return {
        "150": (model_150, scaler_150),
        "250": (model_250, scaler_250)
    }


# -----------------------
# DATASET INFO
# -----------------------
def get_dataset_summary():
    X, y = get_X_y()
    return dataset_summary(X, y)


def get_class_distribution(threshold):
    _, y = get_X_y()
    return class_distribution(y, threshold)


# -----------------------
# PREDICTION
# -----------------------
def predict(models, patient_dict, threshold="250"):
    model, scaler = models[threshold]

    df = pd.DataFrame([patient_dict])

    return predict_patient(df, model, scaler)

if __name__ == '__main__':
    print("\nDATASET SUMMARY:")
    print(get_dataset_summary())

    print("\nCLASS DISTRIBUTION (150):")
    print(get_class_distribution(150))

    print("\nCLASS DISTRIBUTION (250):")
    print(get_class_distribution(250))
