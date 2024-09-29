import numpy as np
import pandas as pd

from .individual import Individual
from mutations import MutatorFactory
from recombinators import NoneCombinator
from .strategy import Strategy
from .strategies import advanced_es


class ESFactory:
    """ TODO:
        - [x] Make sure Weights are between [0, 1)
        - [x] Make sure Weighted Sum are 1 (100%).
        - [x] Create Logger & Log chromosones.
        - [ ] Create Download function.
    
    """
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets= monthly_returns.shape[1] # R^n => R^1
        self.__mean_returns = monthly_returns.mean()
        self.factory = MutatorFactory()

        """ Created an enclosed function! """
        def find_fitness(individual: Individual) -> float:
            fitness = np.dot(self.__mean_returns, individual.chromosone)
            individual.set_fitness(fitness)
            return fitness

        self.fitness_evaluator = find_fitness

    def create_basic(self, steps: int, population_size: int=1, offspring_size: int=1) -> Strategy:
        learning_rate = 1

        # Generate Population
        chromosones = np.random.rand(population_size, self.__n_assets)

        population = [Individual(
            chromosone=chromosones[i],
            mutator=self.factory.create_self_adaptive(
                steps=steps, learning_rate=learning_rate))
                      for i in range(population_size)]
        
        strategy = advanced_es.AdvancedES(
            initial_population=population,
            recombinator=NoneCombinator(),
            evaluator=self.fitness_evaluator,
            offspring_size=offspring_size)

        return strategy


    # def create_advanced(self) -> Portifolio:
    #     pass
