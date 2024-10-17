from abc import ABC, abstractmethod
from typing import List
from ea.individual import Individual

class Recombinator(ABC):
    """Abstract base class for implementing recombination strategies in EAs."""
    @abstractmethod
    def recombinate(self, parents: List[Individual]) -> List[Individual]:
        pass