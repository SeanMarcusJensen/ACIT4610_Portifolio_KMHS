from typing import List
import numpy as np

from mutations.abstraction import MutatorFactory
from .individual import Individual
from evaluators.abstraction import FitnessEvaluator
from recombinators.abstraction import Recombinator
from selections.abstraction import Selector
from utils.logger import Logger

class Strategy:
    def __init__(self,
                 mutator_factory: MutatorFactory,
                 fitness_evaluator: FitnessEvaluator,
                 selector: Selector,
                 recombinator: Recombinator) -> None:
        self.mutator_factory = mutator_factory
        self.fitness_evaluator = fitness_evaluator
        self.selector = selector
        self.recombinator = recombinator

    def fit(self, n_genes: int, n_population: int, n_generations: int, logger: Logger) -> List[Individual]:
        population = self.__create_initial_population(n_genes, n_population)

        for gen in range(n_generations):
            offsprings = self.recombinator.recombinate(population)
            offsprings = [ind.mutate() for ind in offsprings]

            for i in offsprings:
                i.set_fitness(self.fitness_evaluator.evaluate(i)) # evaluate

            population = self.selector.select(parents=population, offsprings=offsprings)

            logger.info(generation=gen, population=population)
        
        return population

    def __create_initial_population(self, n_genes: int, n_population: int) -> List[Individual]:
        return [self.__create_diverse_individual(n_genes) for _ in range(n_population)]

    def __create_diverse_individual(self, n_genes: int) -> Individual:
        chromosone = np.random.dirichlet(np.ones(n_genes) * 0.1)
        individual = Individual(chromosone, self.mutator_factory.create())
        individual.set_fitness(self.fitness_evaluator.evaluate(individual))
        return individual
    