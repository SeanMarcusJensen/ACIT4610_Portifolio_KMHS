import numpy as np
from mutations.abstraction import Mutator

class Individual:
    """ Represents an individual with
        'Real Numbered Continous Representation' in the Evolutionary Strategy.
    """
    def __init__(self, chromosone: np.ndarray, mutator: Mutator) -> None:
        self.chromosone = chromosone
        self.mutator = mutator
        self.fitness = 0.0
    
    def set_fitness(self, value: float) -> None:
        self.fitness = value

    def mutate(self) -> 'Individual':
        """ Mutates the Real Representation.
        〈x1,...,xn〉→〈x′1,...,x′n〉, where xi,x′i ∈ [Li,Ui].
        Li => Lower Bounds (0),
        Ui => Upper Bounds (1),

        this is achieved by adding to the current gene value an amount drawn
        randomly from a Gaussian distribution with mean zero
        and user-specified standard deviation,
        and then curtailing the resulting value to the range [Li,Ui] if necessary.

        Formula:
        p(Δxi)= 1 / σ√2π · e^− (Δxi−ξ)^2 / 2σ^2 .
        """
        self.chromosone = self.mutator.mutate(self.chromosone)
        self.chromosone = self.__normalize_chromosone(self.chromosone)
        self.set_fitness(0.0)
        return self
    
    def copy(self) -> 'Individual':
        return Individual(self.chromosone.copy(), self.mutator.copy())
    
    @staticmethod
    def __normalize_chromosone(chromosone: np.ndarray) -> np.ndarray:
        chromosone = np.clip(chromosone, 0, None)
        chromosone = np.maximum(chromosone, 1e-6)
        sum = chromosone.sum()
        if sum > 0:
            return chromosone / sum
        return np.zeros_like(chromosone)
    
    def normalize_chromosone(self) -> 'Individual':
        self.chromosone = self.__normalize_chromosone(self.chromosone)
        return self
