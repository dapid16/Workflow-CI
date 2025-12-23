import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=1000)
args = parser.parse_args()

# --------------------
# Paths & MLflow
# --------------------
BASE_DIR = Path(__file__).resolve().parent
MLRUNS_DIR = BASE_DIR / "mlruns"

mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("Diabetes_Prediction")

mlflow.autolog()

# --------------------
# Load dataset
# --------------------
data = pd.read_csv(BASE_DIR / "data_preprocessed.csv")

X = data.drop(columns=["diabetes"])
y = data["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------
# Train model
# --------------------
with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=args.max_iter, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head())

    print(f"Run selesai. Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix:\n{cm}")
