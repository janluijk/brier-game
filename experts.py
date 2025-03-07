import random

class Expert:
    def __init__(self, name, description, accuracy=0.5):
        self.weight = 1.0
        self.name = name
        self.description = description
        self.accuracy = accuracy

    def calculate_bounds(self):
        """
        Calculates the lower and upper bounds for accuracy.

        The accuracy 'a' determines the prediction range as follows:
        - a = 0: always returns 0 (worst case).
        - a = 0.5: returns a random value between 0 and 1 (random case).
        - a = 1: always returns 1 (best case).
        """

        lower_bound = max(2 * self.accuracy - 1, 0)
        upper_bound = min(2 * self.accuracy, 1)
        return lower_bound, upper_bound

    def predict(self):
        lower_bound, upper_bound = self.calculate_bounds()
        return random.uniform(lower_bound, upper_bound)
