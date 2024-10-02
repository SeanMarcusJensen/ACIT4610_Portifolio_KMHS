from mutations.abstraction.mutator import Mutator
from mutations.self_adaptive import NStepMutator
from mutations.self_adaptive import OneStepMutator

from mutations.basic_mutator import BasicMutator
from abc import ABC, abstractmethod

class MutatorFactory(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create(self) -> Mutator:
        pass