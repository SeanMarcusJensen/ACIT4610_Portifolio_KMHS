import pandas as pd

from mutations import SelfAdaptiveMutatorFactory, BasicMutatorFactory
from recombinators import NoneCombinator, DiscreteRecombinator, UniformCrossover 
from evaluators import BasicReturnsEvaluator, MaximumReturnsEvaluator, SharpeRatioEvaluator
from selections import TournamentSelector
from es.strategy import Strategy

class EvolutionaryFactory:
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets= len(monthly_returns.columns)# R^n => R^1
    
    def create_basic_es(self, learning_rate: float, recombination_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(recombination_rate=recombination_rate)
        selector = TournamentSelector(n_population=n_population)
        mutator = SelfAdaptiveMutatorFactory(steps=self.__n_assets) # TODO: add Mutation Rate to it. and Abstraction
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)
        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy