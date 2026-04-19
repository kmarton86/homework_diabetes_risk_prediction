import pandas as pd

from dataset import load_data
from model import create_labels, train_model
from prediction import predict_patient
from analysis import dataset_summary, class_distribution

# -----------------------
# LOAD DATA
# -----------------------
X, y = load_data()

# -----------------------
# LABELS
# -----------------------
y_150 = create_labels(y, 150)
y_250 = create_labels(y, 250)

# -----------------------
# TRAIN MODELS
# -----------------------
model_150, scaler_150 = train_model(X, y_150)
model_250, scaler_250 = train_model(X, y_250)

# -----------------------
# ANALYSIS
# -----------------------
print("\nDATASET SUMMARY:")
print(dataset_summary(X, y))

print("\nCLASS DISTRIBUTION (150):")
print(class_distribution(y, 150))

print("\nCLASS DISTRIBUTION (250):")
print(class_distribution(y, 250))

# -----------------------
# NEW PATIENT
# -----------------------
new_patient = {
    "age": 50,
    "sex": 1,
    "bmi": 28,
    "bp": 120,
    "s1": 85,
    "s2": 180,
    "s3": 40,
    "s4": 5,
    "s5": 4.5,
    "s6": 90
}

# -----------------------
# PREPROCESS
# -----------------------
patient_df = pd.DataFrame([new_patient])

# -----------------------
# PREDICTION
# -----------------------
print("\nPREDICTION 150:")
print(predict_patient(patient_df, model_150, scaler_150))

print("\nPREDICTION 250:")
print(predict_patient(patient_df, model_250, scaler_250))