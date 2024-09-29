from abc import ABC, abstractmethod
from typing import List
from es.individual import Individual
from utils.logger import Logger
import numpy as np

class Strategy(ABC):
    def __init__(self, initial_population: List[Individual]) -> None:
        self._population = initial_population
    
    def fit(self, generations: int, logger: Logger) -> List[Individual]:
        for gen in range(generations):
            offsprings = self.create_offsprings()
            new_population = self.select(offsprings)

            logger.log(
                generation=gen,
                sum_chromosone=sum([o.chromosone.sum() for o in new_population]) / len(new_population),
                max_fitness=max([o.fitness for o in new_population]),
                mean_fitness=np.mean([o.fitness for o in new_population]),
                n_offsprings=len(offsprings))

            self._population = new_population
            logger.info(generation=gen, population=new_population)

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
