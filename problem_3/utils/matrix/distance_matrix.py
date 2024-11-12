import numpy as np
import pandas as pd

class DistanceMatrix:
    """Computes the distance matrix for customer locations.
    
    Attributes:
        locations (pd.DataFrame): The DataFrame containing customer locations.
        matrix (np.ndarray): The distance matrix representing distances between customers.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.matrix = self._create_distance_matrix()

    @staticmethod
    def calculate_euclidean_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        return np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)
    
    def _create_distance_matrix(self) -> np.ndarray:
        n_locations = len(self.df)
        distance_matrix = np.zeros((n_locations, n_locations))

        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    distance_matrix[i, j] = self.calculate_euclidean_distance(self.df['Lat'][i], self.df['Lng'][i], self.df['Lat'][j], self.df['Lng'][j])

        return distance_matrix
