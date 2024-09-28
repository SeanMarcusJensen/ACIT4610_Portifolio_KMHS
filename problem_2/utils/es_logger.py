import numpy as np
from utils.logger import Logger


class ESLogger(Logger):
    def __init__(self) -> None:
        self.generations = []
        self.fitness = []

    def info(self, **kwargs) -> None:
        if 'generation' in kwargs and 'population' in kwargs:
            self.generations.append(kwargs['generation'])
            best_fitness = max(individual.fitness for individual in kwargs['population'])
            self.fitness.append(best_fitness)