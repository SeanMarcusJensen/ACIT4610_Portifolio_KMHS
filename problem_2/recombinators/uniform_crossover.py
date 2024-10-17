import numpy as np
from typing import List
from ea.individual import Individual
from recombinators.abstraction import Recombinator

class UniformCrossover(Recombinator):
    """Implements a uniform crossover strategy for creating offspring individuals.

    This recombinator combines pairs of parent individuals using a uniform selection mechanism, where each gene is chosen from one of 
    the parents based on a specified recombination rate.

    Attributes:
        recombination_rate (float): The probability of selecting each gene from the first parent (default is 0.5).
    """
    def __init__(self, recombination_rate: float = 0.5, mutation_probability: float = 0.05):
        self.recombination_rate = recombination_rate
        self.mutation_probability = mutation_probability

    def recombinate(self, population: List[Individual]) -> List[Individual]:
        return [self._recombinate_pair(parent1, parent2) for parent1, parent2 in zip(population[::2], population[1::2])]

    def _recombinate_pair(self, parent1: Individual, parent2: Individual) -> Individual:
        """Performs uniform crossover on a pair of parent individuals to create an offspring.

        A mask is created based on the recombination rate, selecting genes from either parent to form the offspring.

        Args:
            parent1 (Individual): The first parent individual.
            parent2 (Individual): The second parent individual.

        Returns:
            Individual: The newly created offspring individual resulting from the crossover.
        """
        mask = np.random.rand(len(parent1.chromosome)) < self.recombination_rate
        child = np.where(mask, parent1.chromosome.copy(), parent2.chromosome.copy())

        mutation_mask = np.random.rand(len(child)) < self.mutation_probability  # Probability of mutation chance for each gene
        mutator = parent1.mutator.copy()

        # Mutate only the genes selected by the mutation mask
        child[mutation_mask] = mutator.mutate(child[mutation_mask])

        return Individual(child, parent1.mutator.copy())