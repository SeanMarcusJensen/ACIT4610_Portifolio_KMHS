from typing import List
from ea.individual import Individual
from recombinators.abstraction import Recombinator

class NoneCombinator(Recombinator):
    def recombinate(self, parents: List[Individual]) -> List[Individual]:
        offsprings = [ind.copy() for ind in parents]
        return offsprings