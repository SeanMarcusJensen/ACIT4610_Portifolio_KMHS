from typing import List
from selections.abstraction.selector import Selector
from ea.individual import Individual

class ElitismSelector(Selector):
    """Implements an elitism selection strategy for selecting individuals from a population.

    Attributes:
        _n_elites (int): The number of elite individuals to retain in the new population.
        _n_population (int): The number of individuals to retain in the new population (default is 10).
    """
    def __init__(self, n_elites: int = 1, n_population: int = 10) -> None:
        self._n_elites = n_elites
        self._n_population = n_population

    def get_n_population(self) -> int:
        return self._n_population

    def select(self, parents: List[Individual], offsprings: List[Individual]) -> List[Individual]:
        combined_population = parents + offsprings # mu + lambda
        
        # Sort the combined population by fitness in descending order and select the top _n_elites
        sorted_population = sorted(combined_population, key=lambda x: x.fitness, reverse=True)
        new_population = sorted_population[:self._n_elites]
        
        return new_population