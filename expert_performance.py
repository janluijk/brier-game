from typing import Dict, List
import numpy as np

ExpertName = str
ProbabilityDistribution = List[float]
Predictions = Dict[ExpertName, ProbabilityDistribution]

class ExpertPerformanceTracker:
    def __init__(self, experts: List[ExpertName], total_outcomes: int):
        self.experts: List[ExpertName] = experts
        self.weights: Dict[ExpertName, float] = {e: 1.0 for e in experts}
        self.losses: Dict[ExpertName, float] = {e: 0.0 for e in experts}
        self.system_loss: float = 0.0
        self.outcome_space: List[int] = list(range(1, total_outcomes+ 1))

    def brier_score(self, outcome: int, predictions: ProbabilityDistribution) -> float:
        return sum((predictions[o - 1] - (1 if outcome == o else 0)) ** 2 for o in self.outcome_space)

    def update_weights(self, expert_predictions: Predictions) -> None:
        total_weight = 0
        for expert, predictions in expert_predictions.items():
            score = self.brier_score(1, predictions)
            self.weights[expert] *= np.exp(-score)
            total_weight += self.weights[expert]
                       
        if total_weight > 0:
            for expert in self.weights:
                self.weights[expert] /= total_weight

    def update_expert_losses(self, expert_predictions: Predictions) -> None:
        for expert, predictions in expert_predictions.items():
            score = self.brier_score(1, predictions)
            self.losses[expert] += score

    def update_system_loss(self, system_predictions: ProbabilityDistribution) -> None:
        score = self.brier_score(1, system_predictions)
        self.system_loss += score
