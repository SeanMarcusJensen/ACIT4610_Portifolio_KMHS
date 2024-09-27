
from abc import ABC, abstractmethod
from typing import Any, List
import pandas as pd
import numpy as np


class Mutator(ABC):
    def __init__(self, threshold: float = 0.0) -> None:
        self.treshold = threshold

    def mutate(self, representation: np.ndarray) -> np.ndarray:
        """Can be used to create different mutation styles."""
        self._mutate_sigma()
        return self._mutate_objectives(representation)
    
    @abstractmethod
    def _mutate_sigma(self) -> None:
        """Abstract Protected Method"""
        pass
    
    @abstractmethod
    def _mutate_objectives(self, chromosone: np.ndarray) -> np.ndarray:
        """Abstract Protected Method"""
        pass

class OneStepMutator(Mutator):
    def __init__(self, learning_rate: float, threshold: float) -> None:
        super().__init__(threshold=threshold)
        self.learning_rate = learning_rate
        self.sigma = np.random.rand()
    
    def _mutate_sigma(self) -> None:
        # Update the Sigma first!
        sigma_prime = self.sigma * np.exp(self.learning_rate * np.random.normal(0, 1))

        # Apply Threshold to step sizes
        sigma_prime = np.maximum(sigma_prime, self.treshold)
        if sigma_prime < self.treshold:
            sigma_prime = self.treshold
        
        self.sigma = sigma_prime
    
    def _mutate_objectives(self, chromosone: np.ndarray) -> np.ndarray:
        """ Update the objectives in chromosone.
        Formula: x′i = xi + σ′ · Ni(0, 1)
        where Ni(0, 1) is a set of len(x) values of normal values.
        """
        return chromosone + self.sigma * np.random.normal(0, 1, size=chromosone.shape)

class NStepMutatot(Mutator):
    def __init__(self, learning_rate: float, threshold: float, objectives: int) -> None:
        super().__init__(threshold=threshold)
        self.learning_rate = learning_rate
        self.n = objectives
        self.sigma = np.random.rand(self.n)
        self.tau = self.learning_rate / np.sqrt(2 * np.sqrt(self.n))
        self.tau_prime = self.learning_rate / np.sqrt(2 * self.n)
    
    def _mutate_sigma(self) -> None:
        """
        Formula: σ′ i = σi · eτ ′·N(0,1)+τ ·Ni(0,1),
        """
        # Generate random values
        N0 = np.random.normal(0, 1)  # Single random number for all dimensions
        Ni = np.random.normal(0, 1, size=self.n)  # Individual random numbers
        sigma_prime = self.sigma * np.exp(self.tau_prime * N0 + self.tau * Ni)

        # Apply Threshold to step sizes.
        return np.maximum(sigma_prime, self.treshold)
    
    def _mutate_objectives(self, chromosone: np.ndarray) -> np.ndarray:
        """
        Formula: x′i = xi + σ′ i · Ni(0, 1),
        """
        Ni_mutation = np.random.normal(0, 1, size=self.n)
        return chromosone + (self.sigma * Ni_mutation)

class MutationFactory:
    def __init__(self, learning_rate: float, threshold: float, step_size: int) -> None:
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.step_size = step_size

    def create_mutator(self) -> Mutator:
        if self.step_size <= 1:
            return OneStepMutator(self.learning_rate, self.threshold)
        return NStepMutatot(self.learning_rate, self.threshold, self.step_size)
        

class Individual:
    def __init__(self, chromosone_size: int, mutator: Mutator) -> None:
        self.mutator = mutator 
        self.n = chromosone_size
        self.objectives = self.__create_objective_paramas(self.n)
        self.angles = np.ndarray([])
    
    def __create_objective_paramas(self, size: int) -> np.ndarray:
        # Creates a chromosone of [0, 1) with sum of 1. (100%)
        chromosone = np.random.rand(size)
        chromosone /= chromosone.sum()
        return np.array(chromosone)

    @staticmethod
    def create_from(chromosone: np.ndarray, mutator: Mutator) -> 'Individual':
        individual = Individual(0, mutator=mutator)
        individual.objectives = chromosone
        return individual
    
    def copy(self) -> np.ndarray:
        return self.objectives.copy()

    def mutate(self, mutation_rate: float) -> None:
        """Mutation for ES is done by changing value by adding random noice
        drawn from normal / gaussian distribution.

        x'i = xi + N(0, omega)
        • N(0, omega) is a random Gaussian number with a
            mean of zero and standard deviations of omega.

        Key idea:
        • omega is part of the chromosome <x_1,...,x_n, omega_1,..., omega_n>
        • omega is also mutated into omega' (see later how)

        Args:
            mutation_rate (float): _description_
        """

        """ MUTATION RULE!
        This rule resets  after every k iterations by
         =  / c if ps > 1/5
         =  • c if ps < 1/5
         =  if ps = 1/5
        where ps is the % of successful mutations, 0.817  c  1

        Returns:
            _type_: _description_
        """

        if np.random.rand() >= mutation_rate:
            # Was not its time to mutate :D
            return

        # do mutation
        self.objectives = self.mutator.mutate(self.objectives)

    def recombinate(self) -> None:
        pass

    def fitness(self, returns: np.ndarray) -> float:
        weights = self.objectives
        portfolio_returns = np.dot(returns, weights).sum()
        
        # Calculate fitness (e.g., Sharpe ratio or total return)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Using Sharpe ratio as fitness (assuming risk-free rate of 0 for simplicity)
        return mean_return / std_return if std_return != 0 else 0

class ESLogger(ABC):
    def log(self, generation: int, rp: List[np.ndarray]) -> None:
        pass

class ES:
    def __init__(self) -> None:
        self.generations = 20
        self.population_size = 1
        self.offspring_size = 1
        self.mutation_rate = 1.0
        self.chromosone_size: int 
        self.fitness_weights: np.ndarray
        self.logger: ESLogger
        self.mutation_factory: MutationFactory # Need Mutator Factory

    def with_generations(self, size: int) -> 'ES':
        """Sets the number of generations to run the strategy for.

        Args:
            size (int): The number of generations to run for.

        Returns:
            ES: The Evolutionary Strategy.
        """
        self.generations = size
        return self
    
    def with_population(self, size: int) -> 'ES':
        """Sets the number of allowed population to be created in the algorithm.

        Args:
            size (int): The number of population (alpha)

        Returns:
            ES: The Evolutionary Strategy.
        """
        self.population_size = size
        return self
    
    def with_offsprings(self, size: int) -> 'ES':
        """Sets the number of allowed offsprings to be created in the algorithm.

        Args:
            size (int): The number of offsprings (lambda)

        Returns:
            ES: The Evolutionary Strategy.
        """
        self.offspring_size = size
        return self

    def with_mutation(self, rate: float, mutator: MutationFactory) -> 'ES':
        """Sets the odds for mutation.

        Args:
            rate(float): The odds of mutation.

        Returns:
            ES: The Evolutionary Strategy.
        """
        self.mutation_factory = mutator
        self.mutation_rate = rate 
        return self
    
    def with_fitness(self, fitness_weights: np.ndarray) -> 'ES':
        self.fitness_weights = fitness_weights
        return self

    def with_objectives(self, size: int) -> 'ES':
        """Sets the representation for the Chromosone.

        Args:
            rp (RealRepresentation): The representation.

        Returns:
            ES: The Evolutionary Strategy
        """
        self.chromosone_size = size
        return self
    
    def with_logger(self, logger: ESLogger) -> 'ES':
        self.logger = logger
        return self

    def __create_population(self) -> List[Individual]:
        population = list([Individual(self.chromosone_size, self.mutation_factory.create_mutator())
                      for _ in range(self.population_size)])
        return population
    
    def run(self) -> np.ndarray:

        # Create Initial Population
        population = self.__create_population()

        # Run for all generations
        for gen in range(self.generations):
            # Evaluate
            population.sort(key=lambda x: x.fitness(self.fitness_weights), reverse=True)

            # Select

            # Offspring replace parents only if more fit.
            population = population[:self.population_size]
            
            # Create offspring
            offspring = []
            for _ in range(self.offspring_size):
                # Select a parent randomly from the population
                parent_index = np.random.choice(len(population))
                parent = population[parent_index]

                # Create a child by copying the parent
                child = Individual.create_from(parent.copy(), parent.mutator)

                # Mutate the child
                child.mutate(self.mutation_rate)
                offspring.append(child)

            # Add offspring to population
            population.extend(offspring)
            self.logger.log(gen, [i.objectives for i in population])

        
        return np.array([i.objectives for i in population])