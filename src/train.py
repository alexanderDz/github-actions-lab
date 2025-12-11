import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from ingest import load_data
from prep import prepare_data
from utils.metrics import eval_metrics
from ingest import load_config

if __name__ == "__main__":
    cfg = load_config()

    df = load_data(cfg)
    X_train, X_test, y_train, y_test = prepare_data(cfg, df)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=cfg["model"]["params"]["n_estimators"],
            max_depth=cfg["model"]["params"]["max_depth"],
            random_state=cfg["training"]["random_state"]
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = eval_metrics(y_test, preds)
        mlflow.log_metrics(metrics)

        # Log params
        mlflow.log_param("n_estimators", cfg["model"]["params"]["n_estimators"])
        mlflow.log_param("max_depth", cfg["model"]["params"]["max_depth"])

        # Log del modelo — MLflow retorna un ModelInfo con el URI
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=cfg["mlflow"]["registered_model_name"]
        )

        version = model_info.registered_model_version
        name = cfg["mlflow"]["registered_model_name"]

        # Asignar alias "current" a esta versión
        client = MlflowClient()
        client.set_registered_model_alias(
            name=cfg["mlflow"]["registered_model_name"],
            alias="current",
            version=version
        )

        print("Model logged!")
