import pandas as pd

from mutations import SelfAdaptiveMutatorFactory, BasicMutatorFactory
from recombinators import NoneCombinator, DiscreteRecombinator, UniformCrossover 
from evaluators import BasicReturnsEvaluator, MaximumReturnsEvaluator, SharpeRatioEvaluator
from selections import TournamentSelector
from es.strategy import Strategy

class EvolutionaryFactory:
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets = len(monthly_returns.columns)# R^n => R^1
    
    def create_basic_es(self, learning_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = UniformCrossover()
        mutator = BasicMutatorFactory(learning_rate=learning_rate)
        selector = TournamentSelector(n_population=n_population, n_offsprings=n_offsprings)
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_advanced_es_one_step(self, learning_rate: float, recombination_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(recombination_rate=recombination_rate)
        mutator = SelfAdaptiveMutatorFactory(learning_rate=learning_rate, steps=1, mutation_rate=mutation_rate)
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)
        selector = TournamentSelector(n_population=n_population, n_offsprings=n_offsprings)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_advanced_es_n_step(self, learning_rate: float, recombination_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(recombination_rate=recombination_rate)
        mutator = SelfAdaptiveMutatorFactory(learning_rate=learning_rate, steps=self.__n_assets, mutation_rate=mutation_rate)
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)
        selector = TournamentSelector(n_population=n_population, n_offsprings=n_offsprings)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy