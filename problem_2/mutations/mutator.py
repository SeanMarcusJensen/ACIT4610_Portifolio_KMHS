from abc import ABC, abstractmethod
import numpy as np


class Mutator(ABC):
    @abstractmethod
    def mutate(self, chromosone: np.ndarray) -> np.ndarray:
        pass