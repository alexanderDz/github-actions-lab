import pandas as pd
from src.prep import prepare_data

def test_prepare_data():
    df = pd.DataFrame({"x": [1,2,3], "y": [0,1,0]})
    result = prepare_data(df)
    assert len(result) == 4
