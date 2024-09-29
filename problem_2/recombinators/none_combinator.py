from typing import List
from es.individual import Individual
from recombinators.abstraction import Recombinator

class NoneCombinator(Recombinator):
    def recombinate(self, parents: List[Individual]) -> List[Individual]:
        return parents