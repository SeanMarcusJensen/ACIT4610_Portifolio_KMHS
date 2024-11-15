import numpy as np
from typing import List
from selections.abstraction.selector import Selector
from ea.individual import Individual


class TournamentSelector(Selector):
    """Implements a tournament selection strategy for selecting individuals from a population.

    Attributes:
        _tournament_size (int): The number of individuals participating in each tournament (default is 5).
        _n_population (int): The number of individuals to retain in the new population (default is 10).
        _n_offsprings (int): The number of offspring individuals to consider in the selection process (default is 3).
        _selection_strategy (str): The name of the selection strategy used (default is 'mu_plus_lambda'. Alternatively there is 'mu_comma_lambda').
    """

    def __init__(self, tournament_size: int = 5, n_population: int = 10, n_offsprings: int = 3, selection_strategy: str = 'mu_plus_lambda', basic: bool = False) -> None:
        self._tournament_size = tournament_size
        self._n_population = n_population
        self._n_offsprings = n_offsprings
        self._selection_strategy = selection_strategy
        self.__basic = basic

    def get_n_population(self) -> int:
        return self._n_population

    def select(self, parents: List[Individual], offsprings: List[Individual]) -> List[Individual]:
        if self._selection_strategy == 'mu_plus_lambda':
            if self.__basic:
                return offsprings[:self._n_population]
            population = parents[:self._n_population] + \
                offsprings[:self._n_offsprings]  # mu + lambda
        elif self._selection_strategy == 'mu_comma_lambda':
            population = offsprings[:self._n_offsprings]  # mu , lambda
        else:
            raise ValueError(
                "Choose 'mu_plus_lambda' or 'mu_comma_lambda' in the EvolutionaryFactory class.")

        new_population = []
        for _ in range(self._n_population):
            tournament = np.random.choice(
                population, size=self._tournament_size, replace=True)  # type: ignore
            winner = max(tournament, key=lambda x: x.fitness)
            new_population.append(winner)

        return new_population[:self._n_population]
