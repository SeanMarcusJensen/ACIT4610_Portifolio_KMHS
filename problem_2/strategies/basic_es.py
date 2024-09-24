
from abc import ABC
from typing import Any, List
import pandas as pd
import numpy as np


class RealRepresentation:
    """Represents a chromosone for Real Values in ES.
    """
    def __init__(self, objectives: int, each_obj_has_param: bool = False) -> None:
        """Chromosone should consist of three parts:
        Object Variables, x_1, ..., x_n.
        Strategy Parameters, such as:
            Mutation Step Sizes. Omega_1, ..., Omega_n
                - n_omega is usually either 1 or n
                - In the most general case their number n_omega =
                    (n - n_omega/2)(n_omega - 1).
            Rotation Angles:
                - k = n(n-1)/2

        Args:
            size (int): _description_

        Returns:
            RealRepresentation: _description_
        """
        self.omegas = np.random.rand(1 if each_obj_has_param else objectives) # Depending on how we're doing it. lets be basic
        self.objectives = self.__create_objective_paramas(objectives)
        self.angles = np.ndarray([])

    def __create_objective_paramas(self, size: int) -> np.ndarray:
        # Creates a chromosone of [0, 1) with sum of 1. (100%)
        chromosone = np.random.rand(size)
        return self.__normalize(chromosone);
    
    def __normalize(self, list: np.ndarray) -> np.ndarray:
        list /= list.sum()
        return np.array(list)
    
    def mutate(self, learning_rate: float) -> None:
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
        
        omega_prime = self.omegas * np.exp(
            learning_rate
            * np.array([np.random.normal(0, 1)
                        for _ in range(len(self.omegas))]))

        # x’ = x + N(0, omega’)
        x_prime = self.objectives + omega_prime * np.random.normal(0, 1)
        self.omegas = omega_prime
        self.objectives = x_prime
    
    def get_chromosone(self) -> np.ndarray:
        return self.objectives


class Individual:
    def __init__(self, chromosone_size: int) -> None:
        self.chromosone = RealRepresentation(chromosone_size, False)
    
    @staticmethod
    def create_from(chromosone: RealRepresentation) -> 'Individual':
        individual = Individual(0)
        individual.chromosone = chromosone
        return individual
    
    def copy(self) -> RealRepresentation:
        rp = RealRepresentation(0)
        rp.angles = self.chromosone.angles
        rp.objectives = self.chromosone.objectives
        rp.omegas = self.chromosone.omegas
        return rp

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

        if np.random.rand() <= mutation_rate:
            # Was not its time to mutate :D
            return

        # do mutation
        self.chromosone.mutate(0.1) # Learning Rate 0.1

    def recombinate(self) -> None:
        pass

    def fitness(self, cov: pd.DataFrame) -> float:
        weights = self.chromosone.objectives
        portfolio_returns = (cov * weights).sum(axis=1)
        
        # Calculate fitness (e.g., Sharpe ratio or total return)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Using Sharpe ratio as fitness (assuming risk-free rate of 0 for simplicity)
        return mean_return / std_return if std_return != 0 else 0


class BasicES:
    """_summary_

    Args:
        ES (_type_): _description_
    """
    def __init__(self,
                 monthly_returns: pd.DataFrame,
                 generations: int,
                 population_size: int,
                 offspring_size: int,
                 mutation_rate: float) -> None:
        self.cov = monthly_returns 

        """ Evolutionary Strategy Parameter Setup """
        self.population_size = population_size  # (μ)
        self.oppspring_size = offspring_size    # (λ)
        self.mutation_rate = mutation_rate
        self.generations = generations
    
    def __create_population(self) -> List[Individual]:
        chromosone_size = self.cov.shape[0] # Get number of stocks.
        population = list([Individual(chromosone_size)
                      for _ in range(self.population_size)])
        return population

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """_summary_

        Returns:
            Any: _description_
        """

        # Create Initial Population
        population = self.__create_population()

        # Run for all generations
        for gen in range(self.generations):
            print(f"Generation: {gen}.")

            # Evaluate
            population.sort(key=lambda x: x.fitness(self.cov), reverse=True)

            # Select
            population = population[:self.population_size]
            
            # Create offspring
            offspring = []
            for _ in range(self.oppspring_size):
                # Select a parent randomly from the population
                parent_index = np.random.choice(len(population))
                parent = population[parent_index]

                # Create a child by copying the parent
                child = Individual.create_from(parent.copy())

                # Mutate the child
                child.mutate(self.mutation_rate)
                offspring.append(child)

            # Add offspring to population
            population.extend(offspring)