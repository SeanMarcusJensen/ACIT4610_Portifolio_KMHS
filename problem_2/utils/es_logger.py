
from abc import ABC
from typing import List
import numpy as np

class ESLogger(ABC):
    def log(self, generation: int, rp: List[np.ndarray]) -> None:
        pass