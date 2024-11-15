import pandas as pd

from mutations import SelfAdaptiveMutatorFactory, BasicMutatorFactory
from recombinators import NoneCombinator, DiscreteRecombinator, UniformCrossover
from evaluators import BasicReturnsEvaluator, MaximumReturnsEvaluator, SharpeRatioEvaluator
from selections import TournamentSelector, ElitismSelector
from ea.strategy import Strategy


class EvolutionaryFactory:
    """Factory class for creating different evolutionary strategies used for optimizing portfolios based on monthly returns.

    Attributes:
        monthly_returns (pd.DataFrame): A DataFrame containing the monthly returns of the assets.
        __n_assets (int): The number of assets in the monthly_returns DataFrame, used in advanced evolutionary strategies.
    """

    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets = len(monthly_returns.columns)  # R^n => R^1

    def create_basic_ep(self, learning_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = NoneCombinator()
        mutator = BasicMutatorFactory(learning_rate=learning_rate)
        selector = TournamentSelector(
            n_population=n_population, n_offsprings=n_offsprings, selection_strategy='mu_plus_lambda')
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_advanced_ep(self, learning_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = NoneCombinator()
        mutator = SelfAdaptiveMutatorFactory(
            learning_rate=learning_rate, steps=1, mutation_rate=mutation_rate)
        selector = TournamentSelector(
            n_population=n_population, n_offsprings=n_offsprings, selection_strategy='mu_plus_lambda')
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_basic_es(self, learning_rate: float, recombination_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(
            recombination_rate=recombination_rate)
        mutator = BasicMutatorFactory(learning_rate=learning_rate)
        selector = ElitismSelector(n_elites=n_population,
                                   n_population=n_population, n_offsprings=n_offsprings, strategy='mu_comma_lambda')
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_advanced_es_one_step(self, learning_rate: float, recombination_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(
            recombination_rate=recombination_rate)
        mutator = SelfAdaptiveMutatorFactory(
            learning_rate=learning_rate, steps=1, mutation_rate=mutation_rate)
        selector = ElitismSelector(n_elites=n_population,
                                   n_population=n_population, n_offsprings=n_offsprings, strategy='mu_comma_lambda')
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_advanced_es_n_step(self, learning_rate: float, recombination_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(
            recombination_rate=recombination_rate)
        mutator = SelfAdaptiveMutatorFactory(
            learning_rate=learning_rate, steps=self.__n_assets, mutation_rate=mutation_rate)
        selector = ElitismSelector(n_elites=n_population,
                                   n_population=n_population, n_offsprings=n_offsprings, strategy='mu_comma_lambda')
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_mu_plus_lambda(self, learning_rate: float, recombination_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(
            recombination_rate=recombination_rate)
        mutator = SelfAdaptiveMutatorFactory(
            learning_rate=learning_rate, steps=self.__n_assets, mutation_rate=mutation_rate)
        selector = ElitismSelector(n_elites=n_population,
                                   n_population=n_population, n_offsprings=n_offsprings, strategy='mu_plus_lambda')
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy

    def create_mu_comma_lambda(self, learning_rate: float, recombination_rate: float, mutation_rate: float, n_population: int, n_offsprings: int) -> Strategy:
        recombinator = DiscreteRecombinator(
            recombination_rate=recombination_rate)
        mutator = SelfAdaptiveMutatorFactory(
            learning_rate=learning_rate, steps=self.__n_assets, mutation_rate=mutation_rate)
        selector = ElitismSelector(n_elites=n_population,
                                   n_population=n_population, n_offsprings=n_offsprings, strategy='mu_comma_lambda')
        evaluator = MaximumReturnsEvaluator(self.monthly_returns)

        strategy = Strategy(mutator, evaluator, selector, recombinator)
        return strategy
