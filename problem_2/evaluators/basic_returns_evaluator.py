from evaluators.abstraction.fitness_evaluator import FitnessEvaluator
from ea.individual import Individual
import pandas as pd
import numpy as np

class BasicReturnsEvaluator(FitnessEvaluator):
    """Evaluates the fitness of individuals based on the mean returns of their portfolios.

    Attributes:
        monthly_returns (pd.DataFrame): A DataFrame containing the monthly returns of the assets.
    """
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns

    def evaluate(self, individual: Individual) -> float:
        portfolio_returns = (self.monthly_returns * individual.chromosone).sum(axis=1)
        return portfolio_returns.mean()