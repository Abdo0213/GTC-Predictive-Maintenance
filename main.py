from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Load models and scaler
lstm_model = load_model("Model&Scaler/lstm_rul_model.h5")
lstm_scaler = joblib.load("Model&Scaler/scaler_lstm.pkl")

gru_model = load_model("Model&Scaler/gru_rul_model.h5")
gru_scaler = joblib.load("Model&Scaler/scaler_gru.pkl")

rf_model = joblib.load("Model&Scaler/rf_classification_model.pkl")
rf_features = joblib.load("Model&Scaler/rf_features.pkl")

app = FastAPI(title="Predictive Maintenance API")

SEQUENCE_LENGTH = 50

# ✅ Required feature columns (same order as training)
REQUIRED_FEATURES = [
    "op_setting_1", "op_setting_2",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

class SensorInput(BaseModel):
    engine_id: int
    readings: list  # list of sensor dictionaries


# -----------------------------
# Preprocessing for LSTM (RUL)
# -----------------------------
def preprocess_input(df: pd.DataFrame):
    """Validate + preprocess input: scale, pad, and return sequence."""
    missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing_cols}")

    df = df[REQUIRED_FEATURES]
    df_scaled = scaler.transform(df)

    last_seq = df_scaled[-SEQUENCE_LENGTH:]
    if len(last_seq) < SEQUENCE_LENGTH:
        padded = np.zeros((SEQUENCE_LENGTH, len(REQUIRED_FEATURES)))
        padded[-len(last_seq):] = last_seq
        last_seq = padded

    return np.expand_dims(last_seq, axis=0)

# -----------------------------
# Preprocessing for RF (Classification)
# -----------------------------
def classify_preprocessing(df, threshold=30, window=10):
    """
    Recreate the same preprocessing done during training:
    - Compute RUL
    - Compute binary label
    - Compute rolling features
    """
    # 1. Compute RUL (Remaining Useful Life)
    rul_per_engine = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df["engine_id"].map(rul_per_engine) - df["cycle"]

    # 2. Create binary label
    df["label"] = df["RUL"].apply(lambda x: 1 if x <= threshold else 0)

    # 3. Rolling stats for each sensor
    features_to_engineer = [col for col in df.columns if "sensor" in col]
    for feature in features_to_engineer:
        df[f"rolling_mean_{feature}_{window}"] = (
            df.groupby("engine_id")[feature]
              .rolling(window=window)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df[f"rolling_std_{feature}_{window}"] = (
            df.groupby("engine_id")[feature]
              .rolling(window=window)
              .std()
              .reset_index(level=0, drop=True)
        )

    # 4. Drop NaNs from rolling
    df = df.dropna()

    return df

# -----------------------------
# API Endpoints
# -----------------------------
# @app.post("/predict/sensors")
# async def predict_from_sensors(data: SensorInput):
#     df = pd.DataFrame(data.readings)

#     # RUL prediction
#     processed = preprocess_input(df)
#     pred_rul = model.predict(processed)

#     # Classification
#     X_last = classify_preprocessing(df)
#     pred_class = rf_model.predict(X_last)[0]

#     return {
#         "engine_id": data.engine_id,
#         "predicted_RUL": float(pred_rul[0][0]),
#         "failure_risk": int(pred_class)  # 0 = healthy, 1 = failure risk
#     }

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...), engine_id: int = Form(...)):
    try:
        # ------------------ Load File ------------------
        if file.filename.endswith(".txt"):
            df = pd.read_csv(file.file, sep=r"\s+", header=None)
            all_cols = (
                ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
                + [f"sensor_{i}" for i in range(1, 22)]
            )
            df.columns = all_cols[:df.shape[1]]
        else:
            df = pd.read_csv(file.file)

        # If engine_id column is missing, insert the one from the form
        if "engine_id" not in df.columns:
            df.insert(0, "engine_id", engine_id)

        # ------------------ RUL (LSTM) ------------------
        X = df[REQUIRED_FEATURES]
        X_scaled = gru_scaler.transform(X)

        if len(X_scaled) < SEQUENCE_LENGTH:
            padded = np.zeros((SEQUENCE_LENGTH, X_scaled.shape[1]))
            padded[-len(X_scaled):] = X_scaled
            X_scaled = padded
        else:
            X_scaled = X_scaled[-SEQUENCE_LENGTH:]

        X_scaled = np.expand_dims(X_scaled, axis=0)
        prediction = gru_model.predict(X_scaled)
        predicted_rul = float(prediction[0][0])

        # ------------------ Classification (RF) ------------------
        processed_df = classify_preprocessing(df)  # use your earlier function
        # Align features exactly as during training
        X_class = processed_df[rf_features].iloc[[-1]]  # last cycle only
        pred_class = int(rf_model.predict(X_class)[0])

        return {
            "engine_id": engine_id,
            "predicted_RUL": predicted_rul,
            "failure_risk": pred_class,  # 0 = safe, 1 = high risk
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")