import pandas as pd
import os

INPUT_PATH = "data/processed/v2_test.csv"
OUTPUT_PATH = "data/production/day2_data.csv"

os.makedirs("data/production", exist_ok=True)

df = pd.read_csv(INPUT_PATH)

df = df.sample(n=1000, random_state=42)

df["Air temperature [K]"] += 15
df["Process temperature [K]"] += 10
df["Torque [Nm]"] *= 1.25
df["Tool wear [min]"] += 5

df.to_csv(OUTPUT_PATH, index=False)

print("Day-2 dataset created:", OUTPUT_PATH)
