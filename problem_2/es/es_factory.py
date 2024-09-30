import numpy as np
import pandas as pd

from .individual import Individual
from mutations import MutatorFactory
from recombinators import NoneCombinator, DiscreteRecombinator, UniformCrossover 
from .abstraction.strategy import Strategy
from .strategies import advanced_es
from .types import ESType
from evaluators.abstraction.fitness_evaluator import FitnessEvaluator
from evaluators import BasicReturnsEvaluator, MaximumReturnsEvaluator, SharpeRatioEvaluator

class Strategies:
    from recombinators.abstraction import Recombinator
    from selectors.abstraction import Selector

    def __init__(self,
                 mutator_factory: MutatorFactory,
                 fitness_evaluator: FitnessEvaluator,
                 selector: Selector,
                 recombinator: Recombinator) -> None:

        self.mutator_factory = mutator_factory
        self.fitness_evaluator = fitness_evaluator
        self.selector = selector
        self.recombinator = recombinator

    def fit(self, genes: int, n_population: int, generations: int, learning_rate: float=0.1):
        population = [self.__create_diverse_individual(learning_rate, genes) for _ in range(n_population)]
        for i in population:
            i.set_fitness(self.fitness_evaluator.evaluate(i)) # evaluate

        for gen in range(generations):
            offsprings = self.recombinator.recombinate(parents=population)
            offsprings = [ind.mutate() for ind in offsprings]

            for i in offsprings:
                i.set_fitness(self.fitness_evaluator.evaluate(i)) # evaluate

            population= self.selector.select(parents=population, offsprings=offsprings)
        
    
    def __create_diverse_individual(self, learning_rate: float, n_genes: int):
        chromosone = np.random.dirichlet(np.ones(n_genes) * 0.1)
        return Individual(chromosone, self.mutator_factory.create_basic(learning_rate=learning_rate))
    

class ESFactory:
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns
        self.__n_assets= len(monthly_returns.columns)# R^n => R^1
        self.factory = MutatorFactory()
    
    def create(self, es_type: ESType, population_size: int=1, offspring_size: int=1, learning_rate: float=0.1) -> Strategy:
        def create_diverse_individual():
            chromosone = np.random.dirichlet(np.ones(self.__n_assets) * 0.1)
            return Individual(chromosone, self.factory.create_basic(learning_rate=learning_rate))

        initial_population = [create_diverse_individual() for _ in range(population_size)]

        strategy = advanced_es.AdvancedES(
            initial_population=initial_population,
            recombinator=UniformCrossover(),
            evaluator=MaximumReturnsEvaluator(self.monthly_returns),
            offspring_size=offspring_size,
            es_type=es_type)

        return strategy
 
    def create_basic(self, steps: int, population_size: int=1, offspring_size: int=1) -> Strategy:
        def __create_diverse_individual_uniform():
            chromosone = np.ones(self.__n_assets) / self.__n_assets
            return Individual(chromosone, self.factory.create_self_adaptive(steps, learning_rate=0.3))

        initial_population = [__create_diverse_individual_uniform() for _ in range(population_size)]
        
        strategy = advanced_es.AdvancedES(
            initial_population=initial_population,
            recombinator=NoneCombinator(),
            evaluator=BasicReturnsEvaluator(self.monthly_returns),
            offspring_size=offspring_size,
            es_type=ESType.MuPlusLambda)

        return strategy

    def create_advanced(self, es_type: ESType,  steps: int, population_size: int=1, offspring_size: int=1) -> Strategy:
        def __create_diverse_individual_dirichlet():
            chromosone = np.random.dirichlet(np.ones(self.__n_assets) * 0.1)
            return Individual(chromosone, self.factory.create_self_adaptive(steps, learning_rate=0.3))
        initial_population = [__create_diverse_individual_dirichlet() for _ in range(population_size)]
        
        strategy = advanced_es.AdvancedES(
            initial_population=initial_population,
            recombinator=DiscreteRecombinator(),
            evaluator=SharpeRatioEvaluator(self.monthly_returns),
            offspring_size=offspring_size,
            es_type=es_type)

        return strategy


