from evaluators.abstraction.fitness_evaluator import FitnessEvaluator
from ea.individual import Individual
import pandas as pd
import numpy as np

class SharpeRatioEvaluator(FitnessEvaluator):
    """Evaluates the fitness of individuals based on the Sharpe Ratio of their portfolios.

    Attributes:
        monthly_returns (pd.DataFrame): A DataFrame containing the monthly returns of the assets.
    """
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns

    def evaluate(self, individual: Individual) -> float:
        portfolio_returns = (self.monthly_returns * individual.chromosome).sum(axis=1)
        fit = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0  # Sharpe ratio (simplified)
        return fit