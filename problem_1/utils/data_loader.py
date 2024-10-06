import os
import pandas as pd

def load_data(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    road_segments = pd.DataFrame()
    traffic_volumes = pd.DataFrame()

    for file in os.listdir(root):
        if file.lower().startswith("road"):
            road_segments = pd.concat([road_segments, pd.read_csv(os.path.join(root, file))], axis=0)
        else:
            traffic_volumes = pd.concat([traffic_volumes, pd.read_csv(os.path.join(root, file))], axis=0)

    return road_segments, traffic_volumes