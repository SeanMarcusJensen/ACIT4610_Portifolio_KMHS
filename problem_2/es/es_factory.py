import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Callable

from .individual import Individual
from mutations import MutatorFactory
from recombinators import Recombinator, NoneCombinator

class Logger(ABC):
    @abstractmethod
    def info(**kwargs):
        pass

class Strategy(ABC):
    def __init__(self, initial_population: List[Individual], logger: Logger) -> None:
        self._population = initial_population
        self._logger = logger
    
    def fit(self, generations: int) -> List[Individual]:
        for _ in range(generations):
            offsprings = self.create_offsprings()
            new_population = self.select(offsprings)
            self._population = new_population

            #     # Evaluate

            #     # Select
            #     population = np.array(sorted_population[:population_size])

        return self._population
    
    @abstractmethod
    def create_offsprings(self) -> List[Individual]:
        """ Creates Offsprings based on algorithm.
        Can do both Recombination and Mutation.
        Returns:
            List[Individual]: _description_
        """
        pass
    
    @abstractmethod
    def select(self, offsprings: List[Individual]) -> List[Individual]:
        """_summary_

        Args:
            individuals (List[Individual]): _description_

        Returns:
            List[Individual]: _description_
        """
        pass

class AdvancedES(Strategy):
    def __init__(self,
                 initial_population: List[Individual],
                 logger: Logger,
                 recombinator: Recombinator,
                 evaluator: Callable[[Individual], float]) -> None:
        self.recombinator = recombinator
        self.get_fitness = evaluator
        super().__init__(initial_population, logger=logger)
        
    def create_offsprings(self) -> List[Individual]:
        offsprings = self.recombinator.recombinate(self._population)

        # TODO: MUTATE
        return offsprings

    def select(self, offsprings: List[Individual]) -> List[Individual]:
        combined_population = np.hstack((self._population, offsprings))
        ordered_population = sorted(combined_population, key=self.get_fitness, reverse=True)
        return ordered_population

class ESLogger(Logger):
    def __init__(self) -> None:
        self.generations = np.array([])
        self.fitness = np.array([])

    def info(**kwargs):
        return super().info()

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
            return np.dot(self.monthly_returns.mean(), individual.chromosone)

        self.fitness_evaluator = find_fitness

    def create_basic(self) -> Strategy:
        population_size = 1
        offspring_size = 1
        logger = ESLogger()

        # Generate Population
        chromosones = np.random.rand(population_size, self.__n_assets)
        population = [Individual(
            chromosone=chromosones[i],
            mutator=self.factory.create_self_adaptive(steps=1))
                      for i in range(population_size)]
        
        strategy = AdvancedES(
            initial_population=population,
            logger=logger,
            recombinator=NoneCombinator(),
            evaluator=self.fitness_evaluator)

        return strategy


    # def create_advanced(self) -> Portifolio:
    #     pass
