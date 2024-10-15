import numpy as np
from typing import List
from selections.abstraction.selector import Selector
from ea.individual import Individual

class TournamentSelector(Selector):
    """Implements a tournament selection strategy for selecting individuals from a population.

    Attributes:
        _tournament_size (int): The number of individuals participating in each tournament (default is 5).
        _n_population (int): The number of individuals to retain in the new population (default is 1).
        _n_offsprings (int): The number of offspring individuals to consider in the selection process (default is 1).
    """
    def __init__(self, tournament_size: int = 5, n_population: int = 1, n_offsprings: int = 1) -> None:
        self._tournament_size = tournament_size
        self._n_population = n_population
        self._n_offsprings = n_offsprings

    def select(self, parents: List[Individual], offsprings: List[Individual]) -> List[Individual]:
        population = parents[:self._n_population] + offsprings[:self._n_offsprings] # mu + lambda

        new_population = []
        for _ in range(self._n_population):
            tournament = np.random.choice(population, size=self._tournament_size, replace=False) # type: ignore
            winner = max(tournament, key=lambda x: x.fitness)
            new_population.append(winner)

        return new_population[:self._n_population]