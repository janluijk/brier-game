from math import log
import numpy as np
from scipy.optimize import fsolve

class WeightUpdater:
    def __init__(self, experts, learning_rate=0.1):
        """Initialize experts with equal weights."""
        self.experts = experts
        self.learning_rate = learning_rate
        self.weights = {e.name: 1.0 for e in experts}
        self.outcome_space = [0, 1]

    def brier_score(self, prediction, outcome=1):
        """Computes the Brier score: lower is better."""
        return (prediction - outcome) ** 2

    def update(self, scores):
        """
        Update expert weights using the Vovk-Zhdanov Algorithm.
        :param scores: Dictionary {expert_name: Brier score} 
        """

        for name, score in scores.items():
            self.weights[name] *= np.exp(-score)

        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight


    # def get_weighted_prediction(self, expert_predictions):
    #     """Compute the final prediction as a weighted avarge of expert predictions."""
    #
    #     return sum(self.weights[name] * expert_predictions[name] for name in expert_predictions)

    def log_weighted_loss(self, prediction):
        weighted_loss = sum(self.weights[name] * np.exp(- (self.brier_score(prediction, 1))) for name in self.weights)
        return -np.log(weighted_loss)

    def get_optimal_predictions(self, scores):
        def equation(x):
            return sum(max(x - self.log_weighted_loss(outcome), 0) for outcome in self.outcome_space) - 2

        solution = fsolve(equation, x0=1.0)[0]

        predictions = {
            outcome: max(solution - self.log_weighted_loss(outcome), 0) * 0.5 
            for outcome in self.outcome_space
        }

        return predictions
