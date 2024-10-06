import os
import pandas as pd

def load_data(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the data from the given root directory.

    Args:
        root (str): The root directory of the data folder.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The road segments[0] and traffic volumes[1] dataframes.
    """

    road_segments = pd.DataFrame()
    traffic_volumes = pd.DataFrame()

    for file in os.listdir(root):
        if file.lower().startswith("road"):
            road_segments = pd.concat([road_segments, pd.read_csv(os.path.join(root, file))], axis=0)
        else:
            traffic_volumes = pd.concat([traffic_volumes, pd.read_csv(os.path.join(root, file))], axis=0)

    return road_segments, traffic_volumes