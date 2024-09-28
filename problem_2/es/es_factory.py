import numpy as np
import pandas as pd

from .individual import Individual


class Portifolio:
    """Represents a stock portifolio.
    Constraints:
        - The portifolio as the weighted sum of 1: which represents 100%.
    """
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.covariance = monthly_returns.cov()
        self.__monthly_returns = monthly_returns
        self.__n_assets= monthly_returns.shape[1] # R^n => R^1
        self.__asset_names = monthly_returns.columns
        self.__mean_returns = monthly_returns.mean()
    
    def evaluate(self, generations: int) -> np.ndarray:
        # Find best weights
        weights = self.__find_weights(generations=generations)
        # Calculate Risk
        # Return both risk and weights.
        return weights

    def __risk(self) -> pd.DataFrame:
        """Calculates the riskiness of the portifolio.
        TODO: Calculate the risk from weights and covariance.
        """
        return self.covariance

    def __initialize_population(self, mu: int) -> list[Individual]:
        """_summary_
        Args:
            mu (int): Population Size
        """
        chromosones = np.random.rand(mu, self.__n_assets)
        population = [Individual(chromosone=chromosones[i], sigma_size=1)
                      for i in range(mu)]

        """Make sure we have mu number of individuals,
        with chromosone size equals to the number of assets we have.
        """
        assert(len(population) == mu)
        assert(len(chromosones[0]) == self.__n_assets)

        return population

    def __find_weights(self, generations: int) -> np.ndarray:
        population_size = 100
        offspring_size = 700

        # Generate Population
        population = self.__initialize_population(population_size)

        # Run through all generations
        for _ in range(generations):
            # Mutate
            offsprings = [ind.mutate() for ind in population][:offspring_size]

            # Evaluate
            sorted_population = sorted(
                np.hstack((population, offsprings)),
                key=lambda ind: ind.fitness(self.__mean_returns),
                reverse=True)

            # Select
            population = np.array(sorted_population[:population_size])
        
        # Return best portifolio
        return population[0].chromosone


class ESFactory:
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns

    # def create_basic(self) -> Portifolio:
    #     pass

    # def create_advanced(self) -> Portifolio:
    #     pass