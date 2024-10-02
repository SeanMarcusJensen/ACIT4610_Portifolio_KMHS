from mutations.abstraction.mutator import Mutator
from mutations.self_adaptive import NStepMutator
from mutations.self_adaptive import OneStepMutator
from mutations.abstraction import MutatorFactory


class SelfAdaptiveMutatorFactory(MutatorFactory):
    def __init__(self, learning_rate: float, steps: int, mutation_rate: float) -> None:
        self.learning_rate = learning_rate
        self.steps = steps
        self.mutation_rate = mutation_rate

    def create(self) -> Mutator:
        if self.steps <= 1:
            return OneStepMutator(learning_rate=self.learning_rate, mutation_rate=self.mutation_rate)
        return NStepMutator(learning_rate=self.learning_rate, objectives=self.steps, mutation_rate=self.mutation_rate)