import numpy as np
from mutations.abstraction import Mutator

class OneStepMutator(Mutator):
    """Implements a one-step mutation strategy for chromosomes in EAs.

    Attributes:
        learning_rate (float): The learning rate that determines the magnitude of mutations.
        mutation_rate (float): The probability of applying mutation to the chromosome.
        threshold (float): The minimum value allowed for mutated objectives to avoid zero values.
        sigma (float): The mutation step size.
        success_count (int): The count of successful mutations.
        total_count (int): The total count of mutations performed.
        adaptation_interval (int): The interval at which sigma is adapted.
        adaptation_factor (float): The factor used to adjust sigma.
    """
    def __init__(self, learning_rate: float, mutation_rate: float) -> None:
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.sigma = np.random.rand()
        self.threshold = 1e-6

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
        new = OneStepMutator(self.learning_rate, self.mutation_rate)
        new.sigma = self.sigma
        new.success_count = self.success_count
        new.total_count = self.total_count
        new.adaptation_interval = self.adaptation_interval
        new.adaptation_factor = self.adaptation_factor
        return new

    def _mutate_sigma(self) -> None:
        sigma_prime = self.sigma * np.exp(np.random.normal(0, self.learning_rate))
        sigma_prime = np.maximum(sigma_prime, self.threshold)
        self.sigma = sigma_prime
    
    def _mutate_objectives(self, objective: np.ndarray) -> np.ndarray:
        """ Update the objectives in chromosone.
        Formula: x′i = xi + σ′ · Ni(0, 1)
        where Ni(0, 1) is a set of len(x) values of normal values.
        """
        return objective + self.sigma * np.random.normal(0, 1)
    
    def _adapt_sigma(self):
        success_rate = self.success_count / self.adaptation_interval
        if success_rate > 1/5:
            self.sigma *= self.adaptation_factor
        elif success_rate < 1/5:
            self.sigma /= self.adaptation_factor
        self.success_count = 0
        self.total_count = 0