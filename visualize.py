import matplotlib.pyplot as plt

class Results:
    def __init__(self):
        """Initialize storage for tracking simulation data."""
        self.weights_history = [] # Stores weights over time
        self.prediction_history = [] # Stores final weighted predictions
        self.brier_scores = {} # {expert_name: list of scores}

    def log(self, weights, prediciton, scores):
        """Store results from a single round"""
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
        """Plot how expert weights evolve over time."""
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
        """Plot the final weighted prediction over time."""
        plt.plot(self.results.prediction_history, label="Final Weighted Prediction")
        plt.axhline(y=1, color='r', linestyle='--', label="True Outcome")
        plt.xlabel("Rounds")
        plt.ylabel("Prediction Probability")
        plt.title("Final Weighted Prediction Over Time")
        plt.legend()
        plt.show()

    def plot_brier_scores(self):
        """Plot expert Brier scores over time."""
        for expert_name, scores in self.results.brier_scores.items():
            plt.plot(scores, label=expert_name)
        plt.xlabel("Rounds")
        plt.ylabel("Brier score")
        plt.title("Expert Brier Scores Over Time")
        plt.legend()
        plt.show()


