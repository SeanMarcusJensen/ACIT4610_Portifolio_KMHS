import numpy as np
from typing import List
from es.individual import Individual
from .recombinator import Recombinator

class DiscreteRecombinator(Recombinator):
    def __init__(self, recombination_rate: float = 0.7):
        self.recombination_rate = recombination_rate

    def recombinate(self, population: List[Individual]) -> List[Individual]:
        new_population = []
        np.random.shuffle(population)
        
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                parent1, parent2 = population[i], population[i + 1]
                if np.random.random() < self.recombination_rate:
                    child1, child2 = self._discrete_recombination(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1.copy(), parent2.copy()])
            else:
                new_population.append(population[i].copy())
        
        return new_population

    def _discrete_recombination(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        mask = np.random.randint(2, size=len(parent1.chromosone))
        child1_chromosone = np.where(mask, parent1.chromosone, parent2.chromosone)
        child2_chromosone = np.where(mask, parent2.chromosone, parent1.chromosone)
        
        child1 = Individual(self._normalize_chromosone(child1_chromosone), parent1.mutator.copy())
        child2 = Individual(self._normalize_chromosone(child2_chromosone), parent2.mutator.copy())
        
        return child1, child2
    
    @staticmethod
    def _normalize_chromosone(chromosone: np.ndarray) -> np.ndarray:
        chromosone = np.clip(chromosone, 0, None)
        sum = chromosone.sum()
        if sum > 0:
            return chromosone / sum
        return np.zeros_like(chromosone)