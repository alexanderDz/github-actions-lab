import pandas as pd
from sklearn.model_selection import train_test_split
from ingest import load_config

def prepare_data(df):
    cfg = load_config()
    X = df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]

    return train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )
