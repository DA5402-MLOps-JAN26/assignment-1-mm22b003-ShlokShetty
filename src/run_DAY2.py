import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/production/day2_data.csv"

df = pd.read_csv(DATA_PATH)

for _, row in df.iterrows():
    payload = {
        "Type": int(row["Type"]),
        "Air_temperature_K": row["Air temperature [K]"],
        "Process_temperature_K": row["Process temperature [K]"],
        "Rotational_speed_rpm": row["Rotational speed [rpm]"],
        "Torque_Nm": row["Torque [Nm]"],
        "Tool_wear_min": row["Tool wear [min]"]
    }

    requests.post(API_URL, json=payload)

print("Day-2 data sent to API")
