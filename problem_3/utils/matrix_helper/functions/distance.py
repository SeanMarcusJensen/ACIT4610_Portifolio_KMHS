import numpy as np
from pandas.core.series import Series


def calculate_distance(x1: Series, x2: Series) -> float:
    """ Calculate the Euclidean distance between two points.
    Uses [Lat, Lng] to calculate the distance between two points.
    Args:
        x1 (Series): Location Dataframe Row (i).
        x2 (Series): Location Dataframe Row (j).
    """
    return np.sqrt((x1['Lat'] - x2['Lat'])**2 + (x1['Lng'] - x2['Lng'])**2)
