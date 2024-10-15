import numpy as np
from mutations.abstraction import Mutator

class NStepMutator(Mutator):
    """Implements an n-step mutation strategy for chromosomes in EAs.

    Attributes:
        learning_rate (float): The learning rate that determines the magnitude of mutations.
        objectives (int): The number of objectives in the chromosome.
        mutation_rate (float): The probability of applying mutation to the chromosome.
        threshold (float): The minimum value allowed for mutated objectives to avoid zero values.
        sigma (np.ndarray): The mutation step sizes for each objective.
        tau (float): A constant used for adaptive sigma calculations.
        tau_prime (float): A constant used for adaptive sigma calculations.
        success_count (int): The count of successful mutations.
        total_count (int): The total count of mutations performed.
        adaptation_interval (int): The interval at which sigma is adapted.
        adaptation_factor (float): The factor used to adjust sigma.
    """
    def __init__(self, learning_rate: float, objectives: int, mutation_rate: float) -> None:
        self.threshold = 1e-6
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.n = objectives
        self.sigma = np.random.rand(self.n) * 0.8
        self.tau = 1 / np.sqrt(2 * np.sqrt(self.n))
        self.tau_prime = 1 / np.sqrt(2 * self.n)

        self.success_count = 0
        self.total_count = 0
        self.adaptation_interval = 10  # Adapt every 10 mutations
        self.adaptation_factor = 0.817  # (0.817)^4 ≈ 0.5
    
    def mutate(self, chromosone: np.ndarray) -> np.ndarray:
        self.threshold = 1 / np.sqrt(len(chromosone))

        if np.random.rand() > self.mutation_rate:
            return chromosone

        self._mutate_sigma()
        mutated_chromosone = self._mutate_objectives(chromosone)
        self.total_count += 1
        if self.total_count % self.adaptation_interval == 0:
            self._adapt_sigma()

        return mutated_chromosone
    
    def copy(self) -> Mutator:
        new = NStepMutator(self.learning_rate, self.n, self.mutation_rate)
        new.sigma = self.sigma
        new.success_count = self.success_count
        new.total_count = self.total_count
        new.adaptation_interval = self.adaptation_interval
        new.adaptation_factor = self.adaptation_factor
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
    
    def _adapt_sigma(self):
        success_rate = self.success_count / self.adaptation_interval
        if success_rate > 1/5:
            self.sigma *= self.adaptation_factor
        elif success_rate < 1/5:
            self.sigma /= self.adaptation_factor
        self.success_count = 0
        self.total_count = 0