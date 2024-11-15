import numpy as np
from mutations.abstraction import Mutator


class Individual:
    """Represents an individual with 'Real Numbered Continuous Representation' in the Evolutionary Algorithm.

    Attributes:
        chromosome (np.ndarray): The real-numbered chromosome of the individual.
        mutator (Mutator): The mutator used to apply mutations to the chromosome.
        fitness (float): The fitness score of the individual.
    """

    def __init__(self, chromosome: np.ndarray, mutator: Mutator) -> None:
        self.chromosome = chromosome
        self.mutator = mutator
        self.fitness = 0.0

    def set_fitness(self, value: float) -> None:
        self.fitness = value

    def mutate(self) -> 'Individual':
        """ Mutates the Real Representation.
        〈x1,...,xn〉→〈x′1,...,x′n〉, where xi,x′i ∈ [Li,Ui].
        Li => Lower Bounds (0),
        Ui => Upper Bounds (1),

        this is achieved by adding to the current gene value an amount drawn
        randomly from a Gaussian distribution with mean zero
        and user-specified standard deviation,
        and then curtailing the resulting value to the range [Li,Ui] if necessary.

        Formula:
        p(Δxi)= 1 / σ√2π · e^− (Δxi−ξ)^2 / 2σ^2 .
        """
        self.chromosome = self.mutator.mutate(self.chromosome)
        self.chromosome = self.__normalize_chromosome(self.chromosome)
        self.set_fitness(0.0)
        return self

    def copy(self) -> 'Individual':
        ind = Individual(self.chromosome.copy(), self.mutator.copy())
        ind.fitness = self.fitness
        return ind

    @staticmethod
    def __normalize_chromosome(chromosome: np.ndarray) -> np.ndarray:
        """Normalizes the chromosome so that its elements are scaled to sum to 1.

        - The chromosome values are first clipped to a minimum of 0.
        - If the total sum of the chromosome values is greater than 0, each value is divided by the total sum.
        - If the total sum is 0, the method returns an array of zeros.

        Args:
            chromosome (np.ndarray): The chromosome to be normalized.

        Returns:
            np.ndarray: The normalized chromosome.
        """
        chromosome = np.clip(chromosome, 0, None)
        chromosome = np.maximum(chromosome, 1e-6)
        sum = chromosome.sum()
        if sum > 0:
            return chromosome / sum
        return np.zeros_like(chromosome)

    def normalize_chromosome(self) -> 'Individual':
        self.chromosome = self.__normalize_chromosome(self.chromosome)
        return self
