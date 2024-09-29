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
                 evaluator: Callable[[Individual], float],
                 offspring_size: int) -> None:
        self.recombinator = recombinator
        self.get_fitness = evaluator
        self.mu = len(initial_population)
        self.n_offsprings = offspring_size
        super().__init__(initial_population)
        
    def create_offsprings(self) -> List[Individual]:
        """
        TODO: ALL ARE THE SAME ALMOST ALL OF THE TIME.
        Returns:
            List[Individual]: _description_
        """
        population = self.recombinator.recombinate(self._population)

        offsprings = []
        while len(offsprings) < self.n_offsprings:
            for ind in population:
                if np.random.rand() <= 0.3: 
                    offsprings.append(ind.mutate())
                else:
                    offsprings.append(ind)

        print(f"Offspring Size{len(offsprings)}")
        return offsprings[:self.n_offsprings]

    def select(self, offsprings: List[Individual]) -> List[Individual]:
        combined_population = np.hstack((self._population, offsprings))
        ordered_population = sorted(combined_population, key=self.get_fitness, reverse=True)
        print(f"[0]({self.get_fitness(ordered_population[0])}\n[last]({self.get_fitness(ordered_population[self.n_offsprings])}))")
        return ordered_population[:self.mu]