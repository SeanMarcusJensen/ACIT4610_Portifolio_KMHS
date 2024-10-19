from typing import List
from selections.abstraction.selector import Selector
from ea.individual import Individual

class ElitismSelector(Selector):
    """Implements an elitism selection strategy for selecting individuals from a population.

    Attributes:
        _n_elites (int): The number of elite individuals to retain in the new population.
        _n_population (int): The number of individuals to retain in the new population (default is 10).
        _n_offsprings (int): The number of offspring individuals to consider in the selection process (default is 3).
    """
    def __init__(self, n_elites: int = 2, n_population: int = 10, n_offsprings: int = 3) -> None:
        self._n_elites = n_elites
        self._n_population = n_population
        self._n_offsprings = n_offsprings

    def get_n_population(self) -> int:
        return self._n_population

    def select(self, parents: List[Individual], offsprings: List[Individual]) -> List[Individual]:
        population = parents[:self._n_population] + offsprings[:self._n_offsprings] # mu + lambda
        
        # Sort the combined population by fitness in descending order and select the top individuals
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        new_population = sorted_population[:self._n_elites]
        
        return new_population