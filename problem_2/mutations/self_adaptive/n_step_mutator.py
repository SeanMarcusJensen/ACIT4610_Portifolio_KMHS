import numpy as np
import mutations

class NStepMutator(mutations.Mutator):
    def __init__(self, learning_rate: float, objectives: int) -> None:
        self.threshold = 1e8
        self.learning_rate = learning_rate
        self.n = objectives
        self.sigma = np.random.rand(self.n)
        self.tau = 1 / np.sqrt(2 * np.sqrt(self.n))
        self.tau_prime = 1 / np.sqrt(2 * self.n)
    
    def mutate(self, chromosone: np.ndarray) -> np.ndarray:
        self.threshold = 1 / np.sqrt(len(chromosone))
        self._mutate_sigma()
        return self._mutate_objectives(chromosone)
    
    def copy(self) -> mutations.Mutator:
        new = NStepMutator(self.learning_rate, self.n)
        new.sigma = self.sigma
        return new

    def _mutate_sigma(self) -> None:
        """
        Formula: σ′ i = σi · eτ ′·N(0,1)+τ ·Ni(0,1),
        """
        # Generate random values
        N0 = np.random.normal(0, 1)  # Single random number for all dimensions
        Ni = np.random.normal(0, 1, size=self.n)  # Individual random numbers
        sigma_prime = self.sigma * np.exp(self.tau_prime * N0 + self.tau * Ni)

        # Apply Threshold to step sizes.
        self.sigma = np.maximum(sigma_prime, self.threshold)
    
    def _mutate_objectives(self, chromosone: np.ndarray) -> np.ndarray:
        """
        Formula: x′i = xi + σ′ i · Ni(0, 1),
        """
        Ni_mutation = np.random.normal(0, 1, size=self.n)
        return chromosone + (self.sigma * Ni_mutation)