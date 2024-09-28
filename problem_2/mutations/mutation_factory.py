from mutations.mutator import Mutator
from mutations.self_adaptive.n_step_mutator import NStepMutator
from mutations.self_adaptive.one_step_mutator import OneStepMutator

class MutatorFactory:
    def __init__(self) -> None:
        super().__init__()

    def create_self_adaptive(self, steps: int, learning_rate: float) -> Mutator:
        if steps <= 1:
            return OneStepMutator(learning_rate=learning_rate)
        return NStepMutator(learning_rate=learning_rate, objectives=steps)