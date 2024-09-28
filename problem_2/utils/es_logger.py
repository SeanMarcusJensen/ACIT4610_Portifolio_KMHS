from utils.logger import Logger
import matplotlib.pyplot as plt


class ESLogger(Logger):
    def __init__(self) -> None:
        self.generations = []
        self.fitness = []

    def info(self, **kwargs) -> None:
        if 'generation' in kwargs and 'population' in kwargs:
            self.generations.append(kwargs['generation'])
            best_fitness = max(individual.fitness for individual in kwargs['population'])
            self.fitness.append(best_fitness)
    
    def flush(self) -> None:
        plt.plot(self.fitness, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Maximum fitnessscore by generations')
        plt.show()

        self.generations = []
        self.fitness = []