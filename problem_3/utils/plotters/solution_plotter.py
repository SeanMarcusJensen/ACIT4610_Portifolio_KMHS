import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import numpy as np


@staticmethod
def plot_solution(customers_df: pd.DataFrame, solutions: Dict[str, np.ndarray]) -> None:
    plt.title(f'Customer Locations: {len(customers_df - 1)}')

    plt.scatter(customers_df.iloc[0]['Lng'], customers_df.iloc[0]
                ['Lat'], label='Drive Out', color='red', marker='x', linewidth=5)

    plt.scatter(customers_df.iloc[1:]['Lng'],
                customers_df.iloc[1:]['Lat'], label='Customers')

    for k, v in solutions.items():
        # Add Origin to the start and end of the route
        routes = customers_df.iloc[v]
        routes = pd.concat([customers_df.iloc[[0]], routes,
                           customers_df.iloc[[0]]], ignore_index=True)
        plt.plot(routes['Lng'], routes['Lat'], label=f"Route {k}", alpha=0.7)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
