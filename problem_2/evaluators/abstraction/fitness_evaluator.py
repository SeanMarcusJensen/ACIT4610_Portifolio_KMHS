from abc import ABC, abstractmethod
from ea.individual import Individual

class FitnessEvaluator(ABC):
    """Abstract base class for evaluating the fitness of individuals in evolutionary algorithms."""
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        pass
