
import numpy as np
import mutations

class BasicMutator(mutations.Mutator):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.threshold = 1e-6

    def mutate(self, chromosone: np.ndarray) -> np.ndarray:
        self.threshold = 1 / np.sqrt(len(chromosone))
        mutated_chromosone = self._mutate_objectives(chromosone)
        return np.maximum(mutated_chromosone, self.threshold)
            
    def copy(self) -> mutations.Mutator:
        new = BasicMutator(self.learning_rate)
        return new
    
    def _mutate_objectives(self, objective: np.ndarray) -> np.ndarray:
        """ Update the objectives in chromosone.
        Formula: x′i = xi + σ′ · Ni(0, 1)
        where Ni(0, 1) is a set of len(x) values of normal values.
        """
        return objective + np.random.normal(0, 1)