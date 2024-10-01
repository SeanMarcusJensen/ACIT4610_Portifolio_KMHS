import numpy as np
from typing import List
from selections.abstraction import Selector
from es.individual import Individual

class TournamentSelector(Selector):
    def __init__(self, tournament_size: int = 5, n_population: int = 1):
        self._tournament_size = tournament_size
        self._n_population = n_population

    def select(self, parents: List[Individual], offsprings: List[Individual]) -> List[Individual]:
        population = parents + offsprings # mu + lambda
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        new_population = []
        for _ in range(self._n_population):
            tournament = np.random.choice(population, size=self._tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            new_population.append(winner)
        return new_population