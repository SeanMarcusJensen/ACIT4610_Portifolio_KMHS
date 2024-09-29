import numpy as np

class EPBasic:
    def __init__(self, learning_rate: float, objectives: int) -> None:
        self.learning_rate = learning_rate
        self.objectives = objectives
    
    def _create_population(self, population_size: int) -> np.ndarray:
        return np.random.rand(population_size, self.objectives)
    
    def run(self, population: np.ndarray) -> np.ndarray:
        pass
