import numpy as np
import pandas as pd

from .individual import Individual
from mutations import MutatorFactory
from recombinators import NoneCombinator, DiscreteRecombinator, UniformCrossover 
from .abstraction.strategy import Strategy
from .strategies import advanced_es
from .types import ESType

class ESFactory:
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets= len(monthly_returns.columns)# R^n => R^1
        self.__mean_returns = monthly_returns.mean()
        self.factory = MutatorFactory()

        def evaluate_fitness_sharpe(individual: Individual) -> float:
            portfolio_returns = (self.monthly_returns * individual.chromosone).sum(axis=1)
            fit = portfolio_returns.mean() / portfolio_returns.std()  # Sharpe ratio (simplified)
            individual.set_fitness(fit)
            return fit
        
        def evaluate_fitness_returns(individual: Individual) -> float:
            # Calculate portfolio returns
            portfolio_returns = (self.monthly_returns.values * individual.chromosone).sum(axis=1)
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).prod() - 1
            # Convert to annualized returns
            years = len(portfolio_returns) / 12  # Assuming monthly data
            annualized_returns = (1 + cumulative_returns) ** (1 / years) - 1
            
            individual.set_fitness(annualized_returns)
            return annualized_returns

        """ Created an enclosed function! """
        def find_fitness(individual: Individual) -> float:
            fitness = np.dot(self.__mean_returns, individual.chromosone)
            individual.set_fitness(fitness)
            return fitness

        self.fitness_evaluator = evaluate_fitness_returns
    
    def create(self, es_type: ESType, population_size: int=1, offspring_size: int=1, learning_rate: float=0.1) -> Strategy:
        def create_diverse_individual():
            chromosone = np.random.dirichlet(np.ones(self.__n_assets) * 0.5)
            return Individual(chromosone, self.factory.create_basic(learning_rate=learning_rate))

        initial_population = [create_diverse_individual() for _ in range(population_size)]

        def fitness_evaluator(individual: Individual) -> float:
            fitness = self.monthly_returns.dot(individual.chromosone).sum().mean()
            individual.set_fitness(fitness)
            return fitness

        strategy = advanced_es.AdvancedES(
            initial_population=initial_population,
            recombinator=UniformCrossover(),
            evaluator=self.fitness_evaluator,
            offspring_size=offspring_size,
            es_type=es_type)

        return strategy
 
    def create_basic(self, steps: int, population_size: int=1, offspring_size: int=1) -> Strategy:
        """Create a basic ES strategy without recombination.
        TODO: Add not self adaptive mutation.

        Args:
            steps (int): Number of steps to run the ES for. if 1, then it is a one step ES.
            population_size (int, optional): Number of individuals in the population. Defaults to 1.
            offspring_size (int, optional): Number of offspring to create. Defaults to 1.

        Returns:
            Strategy: A basic ES strategy.
        """

        # Generate Population
        def create_diverse_individual():
            chromosone = np.random.dirichlet(np.ones(self.__n_assets) * 0.5)
            return Individual(chromosone, self.factory.create_self_adaptive(steps, learning_rate=0.1))

        initial_population = [create_diverse_individual() for _ in range(population_size)]
        
        strategy = advanced_es.AdvancedES(
            initial_population=initial_population,
            recombinator=NoneCombinator(),
            evaluator=self.fitness_evaluator,
            offspring_size=offspring_size,
            es_type=ESType.MuPlusLambda)

        return strategy

    def create_advanced(self, es_type: ESType,  steps: int, population_size: int=1, offspring_size: int=1) -> Strategy:
        """Create an advanced ES strategy with recombination.

        Args:
            steps (int): Number of steps to run the ES for. if 1, then it is a one step ES.
            population_size (int, optional): Number of individuals in the population. Defaults to 1.
            offspring_size (int, optional): Number of offspring to create. Defaults to 1.

        Returns:
            Strategy: An advanced ES strategy.
        """
        def create_diverse_individual():
            chromosone = np.random.dirichlet(np.ones(self.__n_assets) * 0.5)
            return Individual(chromosone, self.factory.create_self_adaptive(steps, learning_rate=0.1))

        initial_population = [create_diverse_individual() for _ in range(population_size)]
        
        strategy = advanced_es.AdvancedES(
            initial_population=initial_population,
            recombinator=DiscreteRecombinator(),
            evaluator=self.fitness_evaluator,
            offspring_size=offspring_size,
            es_type=es_type)

        return strategy
