from abc import ABC, abstractmethod
from typing import Any, List
import pandas as pd
import numpy as np
from .mutators import Mutator, MutationFactory
from utils import es_logger as logger

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


class ES:
    def __init__(self) -> None:
        self.generations = 20
        self.population_size = 1
        self.offspring_size = 1
        self.mutation_rate = 1.0
        self.chromosone_size: int 
        self.fitness_weights: np.ndarray
        self.logger: logger.ESLogger
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
    
    def with_logger(self, logger: logger.ESLogger) -> 'ES':
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