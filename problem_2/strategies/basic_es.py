
from abc import ABC
from typing import Any
import pandas as pd
import numpy as np

class ES:
    pass


class BasicES(ES):
    """_summary_

    Args:
        ES (_type_): _description_
    """
    def __init__(self, covariance_matrix: pd.DataFrame,
                 generations: int,
                 population_size: int,
                 offspring_size: int,
                 mutation_rate: float) -> None:
        self.cov = covariance_matrix

        """ Evolutionary Strategy Parameter Setup """
        self.population_size = population_size  # (μ)
        self.oppspring_size = offspring_size    # (λ)
        self.mutation_rate = mutation_rate
        self.generations = generations

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """_summary_

        Returns:
            Any: _description_
        """

        # Create Initial Population

        # Evaluate

        # Run for all generations

            # Evaluate

            # Select

            # Mutate and or Crossover

            # Replace Population


class Individual:
    def __init__(self, chromosone_size: int) -> None:
        self.chromosone: np.ndarray = self.__create_chromosone(chromosone_size)

    def mutate(self, mutation_rate: float, mutation_count: int) -> None:
        """Mutation for ES is done by changing value by adding random noice
        drawn from normal / gaussian distribution.

        x'i = xi + N(0, omega)
        • N(0, omega) is a random Gaussian number with a
            mean of zero and standard deviations of omega.

        Key idea:
        • omega is part of the chromosome <x_1,...,x_n, omega_1,..., omega_n>
        • omega is also mutated into omega' (see later how)

        Args:
            mutation_rate (float): _description_
            mutation_count (int): _description_
        """
        if np.random.rand() <= mutation_rate:
            # do mutation
            for i in range(mutation_count):
                chromosone_number = np.random.randint(0, len(self.chromosone) - 1)
                chromosone_value = 
                self.chromosone[chromosone_number] = np.random.rand()


    def recombinate(self) -> None:
        pass

    def fitness(self) -> None:
        pass

    def __create_chromosone(self, size: int) -> np.ndarray:
        """Chromosone should consist of three parts:
        Object Variables, x_1, ..., x_n.
        Strategy Parameters, such as:
            Mutation Step Sizes. Omega_1, ..., Omega_n
                - n_omega is usually either 1 or n
                - In the most general case their number n_omega =
                    (n - n_omega/2)(n_omega - 1).
            Rotation Angles:
                - k = n(n-1)/2

        Args:
            size (int): _description_

        Returns:
            np.ndarray: _description_
        """

        # Creates a chromosone of [0, 1) with sum of 1. (100%)
        chromosone = np.random.rand(size)
        chromosone /= chromosone.sum()
        return np.array(chromosone)
