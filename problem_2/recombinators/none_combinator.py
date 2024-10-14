from typing import List
from ea.individual import Individual
from recombinators.abstraction import Recombinator

class NoneCombinator(Recombinator):
    """Implements a recombination strategy that does not change the parents."""
    def recombinate(self, parents: List[Individual]) -> List[Individual]:
        offsprings = [ind.copy() for ind in parents]
        return offsprings