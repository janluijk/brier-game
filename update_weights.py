import numpy as np

class WeightUpdater:
    def __init__(self, experts, learning_rate=0.1):
        """Initialize experts with equal weights."""
        self.experts = experts
        self.learning_rate = learning_rate
        self.weights = {e.name: 1.0 for e in experts}

    def update(self, scores):
        """
        Update expert weights using the Vovk-Zhdanov Algorithm.
        :param scores: Dictionary {expert_name: Brier score} 
        """

        for name, score in scores.items():
            self.weights[name] *= np.exp(-self.learning_rate * score)

        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight


    def get_weighted_prediction(self, expert_predictions):
        """Compute the final prediction as a weighted avarge of expert predictions."""
        return sum(self.weights[name] * expert_predictions[name] for name in expert_predictions)
