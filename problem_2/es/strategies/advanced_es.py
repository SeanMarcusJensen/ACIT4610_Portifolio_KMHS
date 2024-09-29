from es.strategy import Strategy
from typing import List, Callable
from es.individual import Individual
from utils.logger import Logger
from recombinators import Recombinator
import numpy as np

from es.types import ESType

class AdvancedES(Strategy):
    def __init__(self,
                 initial_population: List[Individual],
                 recombinator: Recombinator,
                 evaluator: Callable[[Individual], float],
                 offspring_size: int,
                 es_type: ESType) -> None:
        self.recombinator = recombinator
        self.get_fitness = evaluator
        self.mu = len(initial_population)
        self.n_offsprings = offspring_size
        self.mutation_rate = 0.5
        self.es_type = es_type
        super().__init__(initial_population)
        
    def create_offsprings(self) -> List[Individual]:
        population = self.recombinator.recombinate(self._population)

        offsprings = []
        while len(offsprings) < self.n_offsprings:
            for ind in population:
                new_ind = ind.copy()
                if np.random.rand() <= self.mutation_rate: 
                    new_ind.mutate()
                offsprings.append(new_ind)

        return offsprings[:self.n_offsprings]

    def select(self, offsprings: List[Individual]) -> List[Individual]:
        combined_population = self._population + offsprings
        for ind in combined_population:
            ind.set_fitness(self.get_fitness(ind))
        
        if self.es_type == ESType.MuPlusLambda:
            selection_pool = self._population + offsprings
        elif self.es_type == ESType.MuCommaLambda:
            selection_pool = offsprings
        else:
            raise ValueError(f"Invalid ES type: {self.es_type}")
        
        # Tournament selection
        new_population = []
        for _ in range(self.mu):
            tournament = np.random.choice(selection_pool, size=5, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            new_population.append(winner)
        
        return new_population