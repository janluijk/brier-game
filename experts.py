import random

class Expert:
    def __init__(self, name):
        self.weight = 1.0
        self.name = name

    def predict(self):
        return random.uniform(0, 1)


class StrongExpert(Expert):
    def predict(self):
        return random.uniform(0.7, 1.0);

class WeakExpert(Expert):
    def predict(self):
        return random.uniform(0.0, 0.3)
