import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(cfg, df):
    X = df.drop(columns=[cfg["data"]["target"]])
    y = df[cfg["data"]["target"]]

    return train_test_split(
        X, y, test_size=cfg["training"]["test_size"], random_state=cfg["training"]["random_state"]
    )
