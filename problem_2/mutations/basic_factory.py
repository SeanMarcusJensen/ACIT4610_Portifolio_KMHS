from mutations.abstraction.mutator import Mutator
from mutations.basic_mutator import BasicMutator
from mutations.abstraction import MutatorFactory

class BasicMutatorFactory(MutatorFactory):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        super().__init__()

    def create(self) -> Mutator:
        return BasicMutator(learning_rate=self.learning_rate)