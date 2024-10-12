from evaluators.abstraction.fitness_evaluator import FitnessEvaluator
from ea.individual import Individual
import pandas as pd
import numpy as np

class MaximumReturnsEvaluator(FitnessEvaluator):
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns

    def evaluate(self, individual: Individual) -> float:
            # Calculate portfolio returns
            portfolio_returns = (self.monthly_returns.values * individual.chromosone).sum(axis=1)
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).prod() - 1
            # Convert to annualized returns
            years = len(portfolio_returns) / 12  # Assuming monthly data
            annualized_returns = (1 + cumulative_returns) ** (1 / years) - 1

            return annualized_returns