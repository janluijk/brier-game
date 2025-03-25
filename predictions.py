from expert_performance import ExpertPerformanceTracker, ProbabilityDistribution
import numpy as np 
from scipy.optimize import root_scalar
from typing import Dict, List

ExpertName = str
ProbabilityDistribution = List[float]
Predictions = Dict[ExpertName, ProbabilityDistribution]

class PredictionOptimizer:
    def __init__(self, expert_manager: ExpertPerformanceTracker):
        self.expert_manager = expert_manager 
        self.outcome_space = self.expert_manager.outcome_space

    def log_weighted_loss(self, outcome: int, expert_predictions: Predictions) -> float:
        weights = self.expert_manager.weights
        brier_score = self.expert_manager.brier_score

        total_weighted_loss = sum(
            weights[expert] * np.exp(- brier_score(outcome, predictions))
            for expert, predictions in expert_predictions.items()
        )
        return -np.log(total_weighted_loss)

    def get_optimal_predictions(self, expert_predictions: Predictions) -> ProbabilityDistribution:

        def loss_equation(x: float) -> float:
            return sum(
                max(x - self.log_weighted_loss(outcome, expert_predictions), 0.0) 
                for outcome in self.outcome_space
            ) - 2

        solution = root_scalar(loss_equation, method="brentq", bracket=(0, 10))        

        if not solution.converged:
            raise ValueError("Root finding did not converge.")


        return [
            max(solution.root - self.log_weighted_loss(outcome, expert_predictions), 0.0) * 0.5
            for outcome in self.outcome_space
        ]
