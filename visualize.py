import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class Results:
    def __init__(self):
        self.weights_history = [] 
        self.prediction_history = [] 
        self.brier_scores = {} 

    def log(self, weights, prediciton, scores):
        self.weights_history.append(weights.copy())
        self.prediction_history.append(prediciton)

        for name, score in scores.items():
            if name not in self.brier_scores:
                self.brier_scores[name] = []
            self.brier_scores[name].append(score)

class Visualizer:
    def __init__(self, results: Results):
        self.results = results 

    def plot_weights(self):
        for expert_name in self.results.weights_history[0].keys():
            plt.plot(
                    [w[expert_name] for w in self.results.weights_history],
                    label=expert_name
            )

        plt.xlabel("Rounds")
        plt.ylabel("Expert Weight")
        plt.title("Expert Weights Over Time")
        plt.legend()
        plt.show()

    def plot_predictions(self):
        max_probabilities = [max(pred.values()) for pred in self.results.prediction_history]

        plt.plot(max_probabilities, label="Maximum Probability Assigned")
        plt.xlabel("Rounds")
        plt.ylabel("Confidence Level")
        plt.title("Prediction Certainty Over Time")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    def plot_brier_scores(self):
        for expert_name, scores in self.results.brier_scores.items():
            plt.plot(scores, label=expert_name)
        plt.xlabel("Rounds")
        plt.ylabel("Brier score")
        plt.title("Expert Brier Scores Over Time")
        plt.legend()
        plt.show()


