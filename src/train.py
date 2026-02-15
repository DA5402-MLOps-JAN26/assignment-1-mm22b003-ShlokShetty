import os
import yaml
import json
import joblib
import subprocess
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def main():
    config = load_config()

    
    data_version = config["data"]["version"]
    model_cfg = config["model"]
    model_version = model_cfg["version"]

    processed_dir = config["data"]["processed_dir"]

   
    train_path = os.path.join(processed_dir, f"{data_version}_train.csv")
    test_path = os.path.join(processed_dir, f"{data_version}_test.csv")

    ensure_dir("models")
    model_path = os.path.join("models", f"model_{model_version}.pkl")
    metadata_path = os.path.join("models", f"model_{model_version}_metadata.json")

    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "Machine failure"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    algorithm = model_cfg["algorithm"]

    if algorithm == "LogisticRegression":
        model = LogisticRegression(
            max_iter=model_cfg["max_iter"],
            random_state=model_cfg["random_state"]
        )

    elif algorithm == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            random_state=model_cfg["random_state"]
        )

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)


    joblib.dump(model, model_path)

    metadata = {
        "model_version": model_version,
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_version": data_version,
        "training_data": train_path,
        "test_data": test_path,
        "git_commit": get_git_commit(),

        "model": {
            "algorithm": algorithm,
            "hyperparameters": model_cfg
        },

        "metrics": {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_precsion": float(test_precision),
            "test_recall": float(test_recall)
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Training completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")


if __name__ == "__main__":
    main()
