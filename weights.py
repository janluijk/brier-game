import numpy as np

class WeightManager:
    def __init__(self, experts, total_outcomes=2):
        self.experts = experts
        self.weights = {e.name: 1.0 for e in experts}
        self.outcome_space = list(range(1, total_outcomes+ 1))

    def brier_score(self, outcome, predictions):
        return sum((predictions[o - 1] - (1 if outcome == o else 0)) ** 2 for o in self.outcome_space)

    def update_weights(self, expert_predictions):
        total_weight = 0
        for expert, predictions in expert_predictions.items():
            score = self.brier_score(1, predictions)
            self.weights[expert] *= np.exp(-score)
            total_weight += self.weights[expert]
                       
        if total_weight > 0:
            for expert in self.weights:
                self.weights[expert] /= total_weight

