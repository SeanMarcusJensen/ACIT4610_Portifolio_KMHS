import numpy as np
from typing import List
from ea.individual import Individual
from recombinators.abstraction import Recombinator

class DiscreteRecombinator(Recombinator):
    """Implements a discrete recombination strategy for creating offspring individuals.

    This recombinator combines two parent individuals to produce offspring based on a specified recombination rate and a uniform selection rate for chromosomes.

    Attributes:
        recombination_rate (float): The probability of performing recombination between parents (default is 0.7).
        uniform_rate (float): The probability of selecting genes from either parent during recombination (default is 0.5).
    """
    def __init__(self, recombination_rate: float = 0.7, uniform_rate: float = 0.5):
        # Initialize the recombinator with a recombination rate (default 0.7)
        self.recombination_rate = recombination_rate
        self.uniform_rate = uniform_rate

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
        """Performs discrete recombination between two parent individuals to create two offspring.

        A mask is created based on the uniform rate, selecting genes from either parent to form the offspring. Small random perturbations are added to the chromosome values.

        Args:
            parent1 (Individual): The first parent individual.
            parent2 (Individual): The second parent individual.

        Returns:
            tuple[Individual, Individual]: A tuple containing the two newly created offspring individuals.
        """
        mask = np.random.random(len(parent1.chromosone)) < self.uniform_rate
        child1_chromosone = np.where(mask, parent1.chromosone, parent2.chromosone)
        child2_chromosone = np.where(mask, parent2.chromosone, parent1.chromosone)
        
        # Add small random perturbations
        perturbation = np.random.normal(0, 0.01, size=len(child1_chromosone))
        child1_chromosone += perturbation
        child2_chromosone += perturbation
        
        child1 = Individual(child1_chromosone, parent1.mutator.copy())
        child2 = Individual(child2_chromosone, parent2.mutator.copy())
        
        return child1, child2