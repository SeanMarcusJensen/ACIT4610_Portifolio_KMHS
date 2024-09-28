from es.strategy import Strategy
from typing import List, Callable
from es.individual import Individual
from utils.logger import Logger
from recombinators import Recombinator
import numpy as np


class AdvancedES(Strategy):
    def __init__(self,
                 initial_population: List[Individual],
                 recombinator: Recombinator,
                 evaluator: Callable[[Individual], float]) -> None:
        self.recombinator = recombinator
        self.get_fitness = evaluator
        self.mu = len(initial_population)
        super().__init__(initial_population)
        
    def create_offsprings(self) -> List[Individual]:
        offsprings = self.recombinator.recombinate(self._population)
        offsprings = [ind.mutate() for ind in offsprings]
        return offsprings[:self.mu]

    def select(self, offsprings: List[Individual]) -> List[Individual]:
        combined_population = np.hstack((self._population, offsprings))
        ordered_population = sorted(combined_population, key=self.get_fitness, reverse=True)
        return ordered_population