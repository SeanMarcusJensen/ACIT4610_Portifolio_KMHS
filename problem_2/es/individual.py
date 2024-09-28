import numpy as np

class Individual:
    """ Represents an individual with
        'Real Numbered Continous Representation' in the Evolutionary Strategy.
    """
    def __init__(self, chromosone: np.ndarray, sigma_size: int) -> None:
        self.chromosone = chromosone
        self.sigma = np.random.rand(sigma_size)[0] # Random value between 0 and 1.
    
    def fitness(self, weights: np.ndarray) -> float:
        """ The chromosome is dot-multiplied with these weights returns to get the expected portfolio return.

        Args:
            weights (np.ndarray): A list of assets mean returns to evaluate the chromosone.
                Must be of same length as the chromosone.

        Returns:
            float: Represents its strength.
        """
        assert(len(weights) == len(self.chromosone))
        return np.dot(weights, self.chromosone)

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

        self.__one_step_mutate()
        self.__normalize_chromosone()
        
        return self

    def __normalize_chromosone(self) -> None:
        """ Keep the chromosone within the constraints.
        """
        self.chromosone /= self.chromosone.sum()

    def __n_step_mutate(self) -> None:
        pass
    
    def __one_step_mutate(self):
        learning_rate = 0.25
        threshold = 1 / np.sqrt(len(self.chromosone))

        sigma_prime = self.sigma * np.exp(learning_rate * np.random.normal(0, 1))
        sigma_prime = np.maximum(sigma_prime, threshold)

        N = np.random.normal(0, 1, size=len(self.chromosone))
        x_prime = self.chromosone + (sigma_prime * N)

        self.sigma = sigma_prime
        self.chromosone = x_prime