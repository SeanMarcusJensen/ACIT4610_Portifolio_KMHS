import numpy as np
from typing import List
from dataclasses import dataclass, field


@dataclass
class EpsilonGreedy:
    """ Epsilon Greedy Algorithm """
    value: float
    decay: float
    min: float
    history: List[float] = field(default_factory=list)

    @property
    def should_be_random(self) -> bool:
        return np.random.rand() < self.value

    def update(self):
        self.value = max(self.min, self.value * self.decay)
        self.history.append(self.value)
