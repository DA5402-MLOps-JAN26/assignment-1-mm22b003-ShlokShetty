import os
import yaml
import pandas as pd
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def update_manifest(manifest_path, entry):
    with open(manifest_path, "a") as f:
        f.write(entry)
        f.write("\n\n")


def main():
    
    config = load_config()

    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    version = config["data"]["version"]
    train_size = config["data_split"]["train_size"]

    ensure_dir(processed_dir)

    
    df = pd.read_csv(raw_path)

    
    drop_cols = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    df = df.drop(columns=drop_cols)

    
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2}) # added in v2 since I had not encoded earlier

    cleaned_path = os.path.join(processed_dir, f"{version}_cleaned.csv")
    df.to_csv(cleaned_path, index=False)

    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]


    day2_path = "data/production/day2_data.csv"  # added in v3 since datadrift went beyond threshold

    if os.path.exists(day2_path):
        day2_df = pd.read_csv(day2_path)

        if not day2_df.empty:
            train_df = pd.concat([train_df, day2_df], ignore_index=True)
 


    train_path = os.path.join(processed_dir, f"{version}_train.csv")
    test_path = os.path.join(processed_dir, f"{version}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    
    manifest_path = os.path.join(processed_dir, "manifest.txt")
    timestamp = datetime.now()

    manifest_entry = f"""
Data Version: {version}
------------------------------
Created on: {timestamp}
Created by script: src/data_prep.py
Input data: {raw_path}

Operations:
- Dropped identifier columns (UDI, Product ID) and other target columns since it is a binary classification problem.
- Encoded 'Type' column as L=0, M=1, H=2.
- Split data chronologically (train size: {train_size})
"""

    if os.path.exists(day2_path) and not day2_df.empty:
        manifest_entry += """
- Day-2 data available
- added samples to training set for retraining.
"""

    manifest_entry += f"""
Outputs:
- Cleaned data: {cleaned_path}
- Training data: {train_path}
- Test data: {test_path}

Target:
- Machine failure
"""

    update_manifest(manifest_path, manifest_entry)

    print("Data preparation completed successfully.")
    print(f"Processed files saved in: {processed_dir}")


if __name__ == "__main__":
    main()
