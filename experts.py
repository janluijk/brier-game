import numpy as np

class Expert:
    def __init__(self, name, total_outcomes):
        self.name = name
        self.outcome_space = list(range(1, total_outcomes + 1))

    def predict(self):
        raise NotImplementedError("Subclasses implement the predit method")

class RandomExpert(Expert):
    """
    Expert with uniform probability distribution 
    """
    def predict(self):
        probabilities = np.array([np.random.rand() for _ in self.outcome_space])
        return probabilities / np.sum(probabilities)

class BiasedExpert(Expert):
    """
    Expert with a bias towards the true outcome
    """
    def __init__(self, name, bias_strength, outcome_space):
        super().__init__(name, outcome_space)
        self.bias_strength = bias_strength

    def predict(self):
        total_outcomes = len(self.outcome_space)
        probabilities = np.full(total_outcomes, (1 - self.bias_strength) / (total_outcomes - 1))
        probabilities[0] = self.bias_strength
        return probabilities
