import numpy as np
import mutations

class OneStepMutator(mutations.Mutator):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.sigma = np.random.rand()
        self.threshold = 1e9

    def mutate(self, chromosone: np.ndarray) -> np.ndarray:
        self.threshold = 1 / np.sqrt(len(chromosone))
        self._mutate_sigma()
        return self._mutate_objectives(chromosone)

    def _mutate_sigma(self) -> None:
        sigma_prime = self.sigma * np.exp(self.learning_rate * np.random.normal(0, 1))
        sigma_prime = np.maximum(sigma_prime, self.threshold)
        self.sigma = sigma_prime
    
    def _mutate_objectives(self, objective: np.ndarray) -> np.ndarray:
        """ Update the objectives in chromosone.
        Formula: x′i = xi + σ′ · Ni(0, 1)
        where Ni(0, 1) is a set of len(x) values of normal values.
        """
        return objective + self.sigma * np.random.normal(0, 1)