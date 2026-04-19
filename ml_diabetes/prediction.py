# prediction.py

def predict_patient(patient_df, model, scaler):
    """
    patient_df: DataFrame (preprocessed)
    """

    X_scaled = scaler.transform(patient_df)

    pred = model.predict(X_scaled)[0]

    return "Endangered" if pred == 1 else "Not Endangered"