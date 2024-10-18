import numpy as np
from mutations.abstraction import Mutator

class BasicMutator(Mutator):
    """Implements a basic mutation strategy for chromosomes in EAs.

    This mutator applies Gaussian noise to the chromosome's objectives, scaled by a learning rate.
    
    Attributes:
        learning_rate (float): The learning rate that determines the magnitude of mutations.
        threshold (float): The minimum value allowed for mutated objectives to avoid zero values.
    """
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.threshold = 1e-8

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        mutated_chromosome = self._mutate_objectives(chromosome)    
        return np.maximum(mutated_chromosome, self.threshold)
            
    def copy(self) -> Mutator:
        new = BasicMutator(self.learning_rate)
        return new
    
    def _mutate_objectives(self, objective: np.ndarray) -> np.ndarray:
        """ Update the objectives in chromosome.
        Formula: x′i = xi + σ′ · Ni(0, 1)
        where Ni(0, 1) is a set of len(x) values of normal values.
        """
        return objective + self.learning_rate * np.random.normal(0, 1, size=objective.shape)