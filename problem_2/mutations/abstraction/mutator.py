from abc import ABC, abstractmethod
import numpy as np


class Mutator(ABC):
    """Abstract base class for implementing mutation strategies in EAs."""
    @abstractmethod
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def copy(self) -> 'Mutator':
        pass
        