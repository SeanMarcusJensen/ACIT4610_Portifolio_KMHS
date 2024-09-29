from mutations.abstraction.mutator import Mutator
from mutations.self_adaptive import NStepMutator
from mutations.self_adaptive import OneStepMutator

from mutations.basic_mutator import BasicMutator

class MutatorFactory:
    def __init__(self) -> None:
        super().__init__()

    def create_basic(self, learning_rate: float) -> Mutator:
        return BasicMutator(learning_rate=learning_rate)

    def create_self_adaptive(self, steps: int, learning_rate: float) -> Mutator:
        if steps <= 1:
            return OneStepMutator(learning_rate=learning_rate)
        return NStepMutator(learning_rate=learning_rate, objectives=steps)