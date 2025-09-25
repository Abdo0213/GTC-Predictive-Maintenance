from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Load model and scaler
model = load_model("Model&Scaler/lstm_rul_model_new.h5")
scaler = joblib.load("Model&Scaler/scaler_new.pkl")

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

def preprocess_input(df: pd.DataFrame):
    """Validate + preprocess input: scale, pad, and return sequence."""
    
    # 1. Ensure required features exist
    missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing_cols}")
    
    # 2. Drop extra columns & keep only required
    df = df[REQUIRED_FEATURES]
    
    # 3. Scale
    df_scaled = scaler.transform(df)
    
    # 4. Take last SEQUENCE_LENGTH rows
    last_seq = df_scaled[-SEQUENCE_LENGTH:]
    
    # 5. Pad if shorter than SEQUENCE_LENGTH
    if len(last_seq) < SEQUENCE_LENGTH:
        padded = np.zeros((SEQUENCE_LENGTH, len(REQUIRED_FEATURES)))
        padded[-len(last_seq):] = last_seq
        last_seq = padded
    
    return np.expand_dims(last_seq, axis=0)


@app.post("/predict/sensors")
async def predict_from_sensors(data: SensorInput):
    df = pd.DataFrame(data.readings)
    processed = preprocess_input(df)
    pred = model.predict(processed)
    return {"engine_id": data.engine_id, "predicted_RUL": float(pred[0][0])}



@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...), engine_id: int = Form(...)):
    try:
        if file.filename.endswith(".txt"):
            df = pd.read_csv(file.file, sep=r"\s+", header=None)
            all_cols = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, df.shape[1]-5+1)]
            df.columns = all_cols[:df.shape[1]]
        else:
            df = pd.read_csv(file.file)

        # df = pd.read_csv(file.file, sep=r"\s+", header=None)  # NASA data often has space-separated values

        # # Assign column names (first 2 = engine_id, cycle; next are op_settings + sensors)
        # all_cols = ["engine_id", "cycle"] + [f"sensor_{i}" for i in range(1, df.shape[1]-1)]
        # df.columns = all_cols[:df.shape[1]]

        # Keep only required features
        missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        X = df[REQUIRED_FEATURES]

        # Scale & reshape
        X_scaled = scaler.transform(X)
        if len(X_scaled) < SEQUENCE_LENGTH:
            padded = np.zeros((SEQUENCE_LENGTH, X_scaled.shape[1]))
            padded[-len(X_scaled):] = X_scaled
            X_scaled = padded
        else:
            X_scaled = X_scaled[-SEQUENCE_LENGTH:]

        X_scaled = np.expand_dims(X_scaled, axis=0)
        prediction = model.predict(X_scaled)
        predicted_rul = float(prediction[0][0])

        return {"engine_id": engine_id, "predicted_RUL": predicted_rul}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")


# def preprocess_input(df: pd.DataFrame):
#     """Validate + preprocess input: scale, pad, and return sequence."""
    
#     # 1. Validate columns
#     missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
#     extra_cols = [col for col in df.columns if col not in REQUIRED_FEATURES]
    
#     if missing_cols:
#         raise HTTPException(status_code=400, detail=f"Missing required features: {missing_cols}")
#     if extra_cols:
#         raise HTTPException(status_code=400, detail=f"Unexpected features provided: {extra_cols}")

#     # 2. Reorder columns
#     df = df[REQUIRED_FEATURES]

#     # 3. Scale
#     df_scaled = scaler.transform(df)

#     # 4. Take last SEQUENCE_LENGTH rows
#     last_seq = df_scaled[-SEQUENCE_LENGTH:]
    
#     # 5. Pad if shorter than SEQUENCE_LENGTH
#     if len(last_seq) < SEQUENCE_LENGTH:
#         padded = np.zeros((SEQUENCE_LENGTH, len(REQUIRED_FEATURES)))
#         padded[-len(last_seq):] = last_seq
#         last_seq = padded

#     return np.expand_dims(last_seq, axis=0)

# @app.post("/predict/file")
# async def predict_file(file: UploadFile = File(...), engine_id: int = Form(...)):
#     try:
#         df = pd.read_csv(file.file)

#         # Ensure required feature columns exist
#         missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
#         if missing_cols:
#             raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

#         # Select only the required features for prediction
#         X = df[REQUIRED_FEATURES]

#         # Scale & reshape
#         X_scaled = scaler.transform(X)
#         sequence_length = 50
#         if len(X_scaled) < sequence_length:
#             padded = np.zeros((sequence_length, X_scaled.shape[1]))
#             padded[-len(X_scaled):] = X_scaled
#             X_scaled = padded
#         else:
#             X_scaled = X_scaled[-sequence_length:]

#         X_scaled = np.expand_dims(X_scaled, axis=0)

#         prediction = model.predict(X_scaled)
#         predicted_rul = float(prediction[0][0])

#         return {"engine_id": engine_id, "predicted_RUL": predicted_rul}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

