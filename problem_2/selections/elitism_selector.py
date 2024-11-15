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

    def __init__(self, n_elites: int = 2, n_population: int = 10, n_offsprings: int = 3, strategy: str = 'mu_plus_lambda') -> None:
        self._n_elites = n_elites
        self._n_population = n_population
        self._n_offsprings = n_offsprings
        self._strategy = strategy

    def get_n_population(self) -> int:
        return self._n_population

    def select(self, parents: List[Individual], offsprings: List[Individual]) -> List[Individual]:
        # Sort parents and offsprings by fitness
        if self._strategy == 'mu_plus_lambda':
            population = sorted(parents + offsprings,
                                key=lambda x: x.fitness, reverse=True)
        elif self._strategy == 'mu_comma_lambda':
            population = sorted(
                offsprings, key=lambda x: x.fitness, reverse=True)
        else:
            raise ValueError(
                "Choose 'mu_plus_lambda' or 'mu_comma_lambda' in the EvolutionaryFactory class.")

        # Keep the top 'n_elites' individuals as elites
        elites = population[:self._n_elites]

        # Fill the rest of the population with the next best individuals until reaching 'n_population'
        new_population = elites + population[self._n_elites:self._n_population]

        return new_population
