import numpy as np
import pandas as pd

from .individual import Individual
from mutations import MutatorFactory
from recombinators import NoneCombinator
from utils.es_logger import ESLogger
from .strategy import Strategy
from .strategies import advanced_es


class ESFactory:
    """ TODO:
        - [ ] Make sure Weights are between [0, 1)
        - [ ] Make sure Weighted Sum are 1 (100%).
        - [ ] Create Download function.
        - [ ] Create Logger & Log chromosones.
    
    """
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets= monthly_returns.shape[1] # R^n => R^1
        self.__mean_returns = monthly_returns.mean()
        self.factory = MutatorFactory()

        """ Created an enclosed function! """
        def find_fitness(individual: Individual) -> float:
            return np.dot(self.__mean_returns, individual.chromosone)

        self.fitness_evaluator = find_fitness

    def create_basic(self) -> Strategy:
        learning_rate = 0.1
        population_size = 1
        offspring_size = 1
        logger = ESLogger()

        # Generate Population
        chromosones = np.random.rand(population_size, self.__n_assets)

        population = [Individual(
            chromosone=chromosones[i],
            mutator=self.factory.create_self_adaptive(
                steps=1, learning_rate=learning_rate))
                      for i in range(population_size)]
        
        strategy = advanced_es.AdvancedES(
            initial_population=population,
            logger=logger,
            recombinator=NoneCombinator(),
            evaluator=self.fitness_evaluator)

        return strategy


    # def create_advanced(self) -> Portifolio:
    #     pass