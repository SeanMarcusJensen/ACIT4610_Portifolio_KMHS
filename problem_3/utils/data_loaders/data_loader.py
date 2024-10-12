import pandas as pd
import os


def read_data(file_name: str) -> pd.DataFrame:
    path = os.path.join(os.getcwd(), 'data', 'raw', file_name)
    data = pd.read_csv(path)
    return data
