import mlflow
import pandas as pd

def predict(model_uri, data):
    model = mlflow.sklearn.load_model(model_uri)
    return model.predict(data)

if __name__ == "__main__":
    # -----------------------------
    # 1. Load config
    # -----------------------------
    config = load_config()
    model_name = cfg["mlflow"]["registered_model_name"]
    model_alias = "current"

    # -----------------------------
    # 2. Load model from registry
    # -----------------------------
    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.pyfunc.load_model(model_uri)

    # -----------------------------
    # 3. Prepare example input data
    # -----------------------------
    data = load_diabetes()
    X = data.data[:5]   # predict for 5 samples

    # -----------------------------
    # 4. Predict
    # -----------------------------
    preds = model.predict(X)
    print("Predictions:", preds)
