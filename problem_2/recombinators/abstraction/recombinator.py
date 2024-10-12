from abc import ABC, abstractmethod
from typing import List
from ea.individual import Individual

class Recombinator(ABC):
    @abstractmethod
    def recombinate(self, parents: List[Individual]) -> List[Individual]:
        pass