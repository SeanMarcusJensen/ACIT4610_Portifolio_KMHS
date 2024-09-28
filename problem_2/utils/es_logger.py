import numpy as np
from utils.logger import Logger


class ESLogger(Logger):
    def __init__(self) -> None:
        self.generations = np.array([])
        self.fitness = np.array([])

    def info(**kwargs):
        return super().info()