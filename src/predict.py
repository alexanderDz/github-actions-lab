import mlflow
import pandas as pd

def predict(model_uri, data):
    model = mlflow.sklearn.load_model(model_uri)
    return model.predict(data)

if __name__ == "__main__":
    df = pd.read_csv("data/raw/data.csv").head(5).drop(columns=["diabetes"])
    predictions = predict("models:/my_model@prod", df)
    print(predictions)
