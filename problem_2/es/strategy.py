from typing import List
import numpy as np

from mutations import MutatorFactory
from .individual import Individual
from evaluators.abstraction import FitnessEvaluator
from recombinators.abstraction import Recombinator
from selections.abstraction import Selector

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

    def fit(self, genes: int, n_population: int, generations: int, learning_rate: float=0.1) -> List[Individual]:
        population = [self.__create_diverse_individual(learning_rate, genes) for _ in range(n_population)]
        for i in population:
            i.set_fitness(self.fitness_evaluator.evaluate(i)) # evaluate

        for gen in range(generations):
            offsprings = self.recombinator.recombinate(population)
            offsprings = [ind.mutate() for ind in offsprings]

            for i in offsprings:
                i.set_fitness(self.fitness_evaluator.evaluate(i)) # evaluate

            population= self.selector.select(parents=population, offsprings=offsprings)
        
        return population

    def __create_diverse_individual(self, learning_rate: float, n_genes: int):
        chromosone = np.random.dirichlet(np.ones(n_genes) * 0.1)
        return Individual(chromosone, self.mutator_factory.create_basic(learning_rate=learning_rate))
    
