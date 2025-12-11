from sklearn import datasets
import pandas as pd
import yaml

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(cfg):
    source = cfg["data"]["source"]

    if source == "sklearn":
        name = cfg["data"]["name"]

        if name == "diabetes":
            ds = datasets.load_diabetes(as_frame=True)
        elif name == "iris":
            ds = datasets.load_iris(as_frame=True)
        elif name == "breast_cancer":
            ds = datasets.load_breast_cancer(as_frame=True)
        else:
            raise ValueError(f"Unknown sklearn dataset: {name}")

        df = ds.frame
        df["target"] = ds.target
        return df

    else:
        raise ValueError("Unknown data source in config")

if __name__ == "__main__":
    df = load_data()
    print(df.head())