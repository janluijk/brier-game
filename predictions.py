from weights import WeightManager
import numpy as np 

class PredictionOptimizer:
    def __init__(self, weight_manager: WeightManager):
        self.weight_manager = weight_manager
        self.outcome_space = self.weight_manager.outcome_space

    def log_weighted_loss(self, outcome, expert_predictions):
        weights = self.weight_manager.weights
        brier_score = self.weight_manager.brier_score

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
