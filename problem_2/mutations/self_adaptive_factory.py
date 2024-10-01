from mutations.abstraction.mutator import Mutator
from mutations.self_adaptive import NStepMutator
from mutations.self_adaptive import OneStepMutator
from mutations.abstraction import MutatorFactory


class SelfAdaptiveMutatorFactory(MutatorFactory):
    def __init__(self, steps: int) -> None:
        self.steps = steps

    def create(self, learning_rate: float) -> Mutator:
        if self.steps <= 1:
            return OneStepMutator(learning_rate=learning_rate)
        return NStepMutator(learning_rate=learning_rate, objectives=self.steps)