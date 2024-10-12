import pandas as pd
import numpy as np

from typing import Tuple, Callable
from pandas.core.series import Series


def create_matrix(matrix_shape: Tuple[int, int], frame: pd.DataFrame, function: Callable[[Series, Series], float]) -> np.ndarray:
    """ Creates a matrix of shape matrix_shape using the function to calculate the values.

    Args:
        matrix_shape (Tuple[int, int]): The shape of the matrix to create.
        frame (pd.DataFrame): The dataframe to use to calculate the values.
        function (Callable[[Series, Series], float]): A method to calculate the values of the matrix in row i and column j.

    Returns:
        np.ndarray: The matrix of shape matrix_shape with values calculated by 'function'.
    """
    matrix = np.zeros(matrix_shape)
    for i, iv in frame.iterrows():
        for j, jv in frame.iterrows():
            matrix[i, j] = function(iv, jv)
    return matrix
