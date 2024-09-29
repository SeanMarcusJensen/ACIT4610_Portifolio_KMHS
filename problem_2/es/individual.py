import numpy as np
from mutations import Mutator

class Individual:
    """ Represents an individual with
        'Real Numbered Continous Representation' in the Evolutionary Strategy.
    """
    def __init__(self, chromosone: np.ndarray, mutator: Mutator) -> None:
        self.chromosone = chromosone
        self.__normalize_chromosone()
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
        return self
    
    def copy(self) -> 'Individual':
        new = Individual(self.chromosone, self.mutator.copy())
        return new

    def __normalize_chromosone(self) -> None:
        """ Keep the chromosone within the constraints.
        """
        self.chromosone = np.clip(self.chromosone, 0, None)
        sum = self.chromosone.sum()
        self.chromosone = np.array([(c / sum) if c > 0 else 0 for c in self.chromosone])