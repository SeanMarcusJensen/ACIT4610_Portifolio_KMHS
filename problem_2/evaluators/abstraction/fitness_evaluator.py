from abc import ABC, abstractmethod
from es.individual import Individual

class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        pass
