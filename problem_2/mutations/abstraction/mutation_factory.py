from mutations.abstraction.mutator import Mutator
from mutations.self_adaptive import NStepMutator
from mutations.self_adaptive import OneStepMutator

from mutations.basic_mutator import BasicMutator
from abc import ABC, abstractmethod

class MutatorFactory(ABC):
    """Abstract factory class for creating mutators used in EAs."""
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create(self) -> Mutator:
        pass