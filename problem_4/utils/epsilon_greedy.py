import numpy as np
from typing import List
from dataclasses import dataclass, field


@dataclass
class EpsilonGreedy:
    """ Epsilon Greedy Algorithm """
    value: float
    decay: float
    min: float

    @property
    def should_be_random(self) -> bool:
        return np.random.rand() < self.value

    def update(self) -> float:
        self.value = max(self.min, self.value * self.decay)
        return self.value
