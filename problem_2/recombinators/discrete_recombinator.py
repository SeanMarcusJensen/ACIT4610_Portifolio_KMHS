import numpy as np
from typing import List
from es.individual import Individual
from recombinators.abstraction import Recombinator

class DiscreteRecombinator(Recombinator):
    def __init__(self, recombination_rate: float = 0.7):
        # Initialize the recombinator with a recombination rate (default 0.7)
        self.recombination_rate = recombination_rate

    def recombinate(self, population: List[Individual]) -> List[Individual]:
        # This method performs recombination on the entire population
        new_population = []
        np.random.shuffle(population)  # Shuffle the population randomly
        
        # Iterate through the population, taking two individuals at a time
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                # If we have a pair of individuals
                parent1, parent2 = population[i], population[i + 1]
                if np.random.random() < self.recombination_rate:
                    # Perform recombination with probability equal to recombination_rate
                    child1, child2 = self._discrete_recombination(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    # If no recombination, copy the parents
                    new_population.extend([parent1.copy(), parent2.copy()])
            else:
                # If there's an odd number of individuals, copy the last one
                new_population.append(population[i].copy())
        
        return new_population

    def _discrete_recombination(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        # This method performs discrete recombination between two parents
        # Create a random binary mask
        mask = np.random.randint(2, size=len(parent1.chromosone))
        # Create child chromosomes by selecting genes from parents based on the mask
        child1_chromosone = np.where(mask, parent1.chromosone, parent2.chromosone)
        child2_chromosone = np.where(mask, parent2.chromosone, parent1.chromosone)
        
        # Create new Individual objects for the children
        child1 = Individual(self._normalize_chromosone(child1_chromosone), parent1.mutator.copy())
        child2 = Individual(self._normalize_chromosone(child2_chromosone), parent2.mutator.copy())
        
        return child1, child2
    
    @staticmethod
    def _normalize_chromosone(chromosone: np.ndarray) -> np.ndarray:
        # This method normalizes the chromosome to ensure it represents a valid portfolio
        chromosone = np.clip(chromosone, 0, None)  # Ensure all values are non-negative
        sum = chromosone.sum()
        if sum > 0:
            return chromosone / sum  # Normalize so that the sum is 1
        return np.zeros_like(chromosone)  # If all values are 0, return an array of zeros