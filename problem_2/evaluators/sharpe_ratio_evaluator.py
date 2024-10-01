from evaluators.abstraction.fitness_evaluator import FitnessEvaluator
from es.individual import Individual
import pandas as pd
import numpy as np

class SharpeRatioEvaluator(FitnessEvaluator):
    def __init__(self, monthly_returns: pd.DataFrame) -> None:
        self.monthly_returns = monthly_returns

    def evaluate(self, individual: Individual) -> float:
        portfolio_returns = (self.monthly_returns * individual.chromosone).sum(axis=1)
        fit = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0  # Sharpe ratio (simplified)
        return fit