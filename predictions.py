from expert_performance import ExpertPerformanceTracker
import numpy as np 
from scipy.optimize import fsolve

class PredictionOptimizer:
    def __init__(self, expert_manager: ExpertPerformanceTracker):
        self.expert_manager = expert_manager 
        self.outcome_space = self.expert_manager.outcome_space

    def log_weighted_loss(self, outcome, expert_predictions):
        weights = self.expert_manager.weights
        brier_score = self.expert_manager.brier_score

        total_weighted_loss = sum(
            weights[expert] * np.exp(-brier_score(outcome, predictions))
            for expert, predictions in expert_predictions.items()
        )
        return -np.log(total_weighted_loss)

    def get_optimal_predictions(self, expert_predictions):
        def loss_equation(x):
            return sum(
                max(x - self.log_weighted_loss(outcome, expert_predictions), 0.0) 
                for outcome in self.outcome_space
            ) - 2

        solution = fsolve(loss_equation, x0=1.0)[0]

        return [
            max(solution - self.log_weighted_loss(outcome, expert_predictions), 0.0) * 0.5 
            for outcome in self.outcome_space
        ]
