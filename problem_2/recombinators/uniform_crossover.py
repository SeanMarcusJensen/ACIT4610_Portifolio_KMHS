import numpy as np
from typing import List
from ea.individual import Individual
from recombinators.abstraction import Recombinator

class UniformCrossover(Recombinator):
    def __init__(self, recombination_rate: float = 0.7):
        self.recombination_rate = recombination_rate

    def recombinate(self, population: List[Individual]) -> List[Individual]:
        return [self._recombinate_pair(parent1, parent2) for parent1, parent2 in zip(population[::2], population[1::2])]

    def _recombinate_pair(self, parent1: Individual, parent2: Individual) -> Individual:
        mask = np.random.rand(len(parent1.chromosone)) < self.recombination_rate
        child = np.where(mask, parent1.chromosone.copy(), parent2.chromosone.copy())
        return Individual(child, parent1.mutator.copy()) # TODO: mask the mutation part aswell.