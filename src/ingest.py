import pandas as pd
import yaml

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data():
    cfg = load_config()
    df = pd.read_csv(cfg["data_path"])
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())