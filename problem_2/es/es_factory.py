import numpy as np
import pandas as pd

from .individual import Individual
from mutations import MutatorFactory
from recombinators import NoneCombinator, DiscreteRecombinator
from .strategy import Strategy
from .strategies import advanced_es

class ESFactory:
    """ TODO:
        - [x] Make sure Weights are between [0, 1)
        - [x] Make sure Weighted Sum are 1 (100%).
        - [x] Create Logger & Log chromosones.
        - [ ] Create Download function.
    
    """
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets= len(monthly_returns.columns)# R^n => R^1
        self.__mean_returns = monthly_returns.mean()
        self.factory = MutatorFactory()

        def evaluate_fitness_sharpe(individual: Individual) -> float:
            portfolio_returns = (self.monthly_returns * individual.chromosone).sum(axis=1)
            return portfolio_returns.mean() / portfolio_returns.std()  # Sharpe ratio (simplified)
        
        def evaluate_fitness_returns(individual: Individual) -> float:
            # Calculate portfolio returns
            portfolio_returns = (self.monthly_returns.values * individual.chromosone).sum(axis=1)
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).prod() - 1
            # Convert to annualized returns
            years = len(portfolio_returns) / 12  # Assuming monthly data
            annualized_returns = (1 + cumulative_returns) ** (1 / years) - 1
            
            return annualized_returns

        """ Created an enclosed function! """
        def find_fitness(individual: Individual) -> float:
            fitness = np.dot(self.__mean_returns, individual.chromosone)
            individual.set_fitness(fitness)
            return fitness

        self.fitness_evaluator = evaluate_fitness_returns

    def create_basic(self, steps: int, population_size: int=1, offspring_size: int=1) -> Strategy:
        # Generate Population
        def create_diverse_individual():
            chromosone = np.random.dirichlet(np.ones(self.__n_assets) * 0.5)
            return Individual(chromosone, self.factory.create_self_adaptive(steps, learning_rate=0.1))

        initial_population = [create_diverse_individual() for _ in range(population_size)]
        
        strategy = advanced_es.AdvancedES(
            initial_population=initial_population,
            recombinator=DiscreteRecombinator(),
            evaluator=self.fitness_evaluator,
            offspring_size=offspring_size)

        return strategy


    # def create_advanced(self) -> Portifolio:
    #     pass
