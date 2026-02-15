import json
import yaml
import pandas as pd
from sklearn.metrics import recall_score


CONFIG_PATH = "config.yaml"
PREDICTION_LOG = "prediction_log.json"
DAY2_DATA = "data/production/day2_data.csv"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Load config
config = load_config()

MODEL_VERSION = config["deployment"]["active_model_version"]
ERROR_THRESHOLD = config["monitoring"]["error_threshold"]
RECALL_DROP_THRESHOLD = config["monitoring"]["recall_drop_threshold"]

MODEL_METADATA = f"models/model_{MODEL_VERSION}_metadata.json"



with open(MODEL_METADATA, "r") as f:
    metadata = json.load(f)

TRAIN_ACCURACY = metadata["metrics"]["test_accuracy"]
TRAIN_ERROR = 1 - TRAIN_ACCURACY
TRAIN_RECALL = metadata["metrics"]["test_recall"]



with open(PREDICTION_LOG, "r") as f:
    predictions = json.load(f)

pred_df = pd.DataFrame(predictions)



day2_df = pd.read_csv(DAY2_DATA)


n_preds = len(day2_df)
pred_df = pred_df.tail(n_preds)

y_pred = pred_df["prediction"].values
y_true = day2_df["Machine failure"].values



production_error = (y_true != y_pred).mean()
production_recall = recall_score(y_true, y_pred)


print(f"Active model version : {MODEL_VERSION}")
print(f"Baseline error       : {TRAIN_ERROR:.4f}")
print(f"Production error     : {production_error:.4f}")
print(f"Baseline recall      : {TRAIN_RECALL:.4f}")
print(f"Production recall    : {production_recall:.4f}")


# Degradation checks
error_degraded = (production_error - TRAIN_ERROR) > ERROR_THRESHOLD
recall_degraded = production_recall < (TRAIN_RECALL - RECALL_DROP_THRESHOLD)


if error_degraded or recall_degraded:
    print("RETRAINING REQUIRED: performance degradation detected")
else:
    print("Production performance acceptable")
