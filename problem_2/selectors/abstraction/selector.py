from abc import ABC, abstractmethod
from typing import List

from es.individual import Individual

class Selector(ABC):
    @abstractmethod
    def select(self, parents: List[Individual], offsprings: List[Individual]) -> List[Individual]:
        pass