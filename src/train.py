import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from ingest import load_data
from prep import prepare_data
from utils.metrics import eval_metrics
from ingest import load_config

if __name__ == "__main__":
    cfg = load_config()

    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"]
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = eval_metrics(y_test, preds)
        mlflow.log_metrics(metrics)

        # Log params
        for k, v in cfg["model"].items():
            mlflow.log_param(k, v)

        # Log model with signature
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=X_train.iloc[:2],
        )

        print("Model logged!")
