import os
import yaml
import json
import joblib
from datetime import datetime
import csv

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

MODEL_VERSION = config["deployment"]["active_model_version"]
THRESHOLD = config["deployment"]["threshold"]
PORT = config["deployment"]["port"]

MODEL_PATH = os.path.join("models", f"model_{MODEL_VERSION}.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")



app = FastAPI(
    title="Predictive Maintenance Inference API",
    description="Threshold-based inference for machine failure prediction",
    version="1.0"
)


class MachineInput(BaseModel):
    Type: int
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "active_model_version": MODEL_VERSION,
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict(data: MachineInput):
    input_df = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.Air_temperature_K,
        "Process temperature [K]": data.Process_temperature_K,
        "Rotational speed [rpm]": data.Rotational_speed_rpm,
        "Torque [Nm]": data.Torque_Nm,
        "Tool wear [min]": data.Tool_wear_min
    }])

    # Probability of failure (class = 1)
    failure_prob = model.predict_proba(input_df)[0][1]

    prediction = int(failure_prob >= THRESHOLD)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": MODEL_VERSION,
        "inputs": {
            "Type": data.Type,
            "Air_temperature_K": data.Air_temperature_K,
            "Process_temperature_K": data.Process_temperature_K,
            "Rotational_speed_rpm": data.Rotational_speed_rpm,
            "Torque_Nm": data.Torque_Nm,
            "Tool_wear_min": data.Tool_wear_min
        },
        "failure_probability": float(failure_prob),
        "prediction": prediction
    }

    log_path = "prediction_log.json"

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)


    return {
        "model_version": MODEL_VERSION,
        "failure_probability": round(float(failure_prob), 4),
        "threshold": THRESHOLD,
        "prediction": prediction
    }

def log_deployment(status):
    log_path = "deployment_log.csv"
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)

        
        if not file_exists:
            writer.writerow([
                "timestamp",
                "model_version",
                "model_path",
                "threshold",
                "port",
                "status"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            MODEL_VERSION,
            MODEL_PATH,
            THRESHOLD,
            PORT,
            status
        ])



try: 
    model = joblib.load(MODEL_PATH) 
    log_deployment(status="STARTED") 
except Exception: 
    log_deployment(status="FAILED") 
    raise

