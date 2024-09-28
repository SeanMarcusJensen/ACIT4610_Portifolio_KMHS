from mutations.mutator import Mutator
from mutations.self_adaptive.n_step_mutator import NStepMutator
from mutations.self_adaptive.one_step_mutator import OneStepMutator

class MutatorFactory:
    def __init__(self) -> None:
        super().__init__()

    def create_self_adaptive(self, steps: int) -> Mutator:
        if steps <= 1:
            return OneStepMutator(learning_rate=0.1)
        return NStepMutator(learning_rate=0.1, objectives=steps)