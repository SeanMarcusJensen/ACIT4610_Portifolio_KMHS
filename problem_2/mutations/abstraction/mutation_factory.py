from mutations.abstraction.mutator import Mutator
from mutations.self_adaptive import NStepMutator
from mutations.self_adaptive import OneStepMutator

from mutations.basic_mutator import BasicMutator
from abc import ABC, abstractmethod

class MutatorFactory(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create(self, learning_rate: float) -> Mutator:
        pass


class SelfAdaptiveMutatorFactory(MutatorFactory):
    def __init__(self, steps: int) -> None:
        self.steps = steps

    def create(self, learning_rate: float) -> Mutator:
        if self.steps <= 1:
            return OneStepMutator(learning_rate=learning_rate)
        return NStepMutator(learning_rate=learning_rate, objectives=self.steps)


class BasicMutatorFactory(MutatorFactory):
    def __init__(self) -> None:
        super().__init__()

    def create(self, learning_rate: float) -> Mutator:
        return BasicMutator(learning_rate=learning_rate)