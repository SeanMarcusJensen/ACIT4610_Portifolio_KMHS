from abc import ABC, abstractmethod
from typing import List
from es.individual import Individual
from utils.logger import Logger


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
