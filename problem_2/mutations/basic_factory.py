from mutations.abstraction.mutator import Mutator
from mutations.basic_mutator import BasicMutator
from mutations.abstraction import MutatorFactory

class BasicMutatorFactory(MutatorFactory):
    def __init__(self) -> None:
        super().__init__()

    def create(self, learning_rate: float) -> Mutator:
        return BasicMutator(learning_rate=learning_rate)