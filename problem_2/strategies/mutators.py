from abc import ABC, abstractmethod
from typing import Any, List
import pandas as pd
import numpy as np

class Mutator(ABC):
    def __init__(self, threshold: float = 0.0) -> None:
        self.treshold = threshold

    def mutate(self, representation: np.ndarray) -> np.ndarray:
        """Can be used to create different mutation styles."""
        self._mutate_sigma()
        return self._mutate_objectives(representation)
    
    @abstractmethod
    def _mutate_sigma(self) -> None:
        """Abstract Protected Method"""
        pass
    
    @abstractmethod
    def _mutate_objectives(self, chromosone: np.ndarray) -> np.ndarray:
        """Abstract Protected Method"""
        pass

class OneStepMutator(Mutator):
    def __init__(self, learning_rate: float, threshold: float) -> None:
        super().__init__(threshold=threshold)
        self.learning_rate = learning_rate
        self.sigma = np.random.rand()
    
    def _mutate_sigma(self) -> None:
        # Update the Sigma first!
        sigma_prime = self.sigma * np.exp(self.learning_rate * np.random.normal(0, 1))

        # Apply Threshold to step sizes
        sigma_prime = np.maximum(sigma_prime, self.treshold)
        self.sigma = sigma_prime
    
    def _mutate_objectives(self, objective: np.ndarray) -> np.ndarray:
        """ Update the objectives in chromosone.
        Formula: x′i = xi + σ′ · Ni(0, 1)
        where Ni(0, 1) is a set of len(x) values of normal values.
        """
        return objective + self.sigma * np.random.normal(0, 1)

class NStepMutatot(Mutator):
    def __init__(self, learning_rate: float, threshold: float, objectives: int) -> None:
        super().__init__(threshold=threshold)
        self.learning_rate = learning_rate
        self.n = objectives
        self.sigma = np.random.rand(self.n)
        self.tau = self.learning_rate / np.sqrt(2 * np.sqrt(self.n))
        self.tau_prime = self.learning_rate / np.sqrt(2 * self.n)
    
    def _mutate_sigma(self) -> None:
        """
        Formula: σ′ i = σi · eτ ′·N(0,1)+τ ·Ni(0,1),
        """
        # Generate random values
        N0 = np.random.normal(0, 1)  # Single random number for all dimensions
        Ni = np.random.normal(0, 1, size=self.n)  # Individual random numbers
        sigma_prime = self.sigma * np.exp(self.tau_prime * N0 + self.tau * Ni)

        # Apply Threshold to step sizes.
        return np.maximum(sigma_prime, self.treshold)
    
    def _mutate_objectives(self, chromosone: np.ndarray) -> np.ndarray:
        """
        Formula: x′i = xi + σ′ i · Ni(0, 1),
        """
        Ni_mutation = np.random.normal(0, 1, size=self.n)
        return chromosone + (self.sigma * Ni_mutation)

class MutationFactory:
    def __init__(self, learning_rate: float, threshold: float, step_size: int) -> None:
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.step_size = step_size

    def create_mutator(self) -> Mutator:
        if self.step_size <= 1:
            return OneStepMutator(self.learning_rate, self.threshold)
        return NStepMutatot(self.learning_rate, self.threshold, self.step_size)
        