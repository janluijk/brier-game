import matplotlib.pyplot as plt
import numpy as np

class Results:
    def __init__(self):
        self.weights_history = []  
        self.losses_history = []  
        self.system_loss_history = []  
        self.prediction_history = []  
        self.system_prediction_history = []
        self.regret_history = []

    def log(self, weights, expert_predictions, expert_losses, system_loss, system_prediction, regret):
        self.weights_history.append({expert: weight for expert, weight in weights.items()})
        self.losses_history.append({expert: loss for expert, loss in expert_losses.items()})
        self.system_loss_history.append(system_loss)
        self.prediction_history.append(expert_predictions)
        self.system_prediction_history.append(system_prediction)
        self.regret_history.append(regret)

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

    def plot_losses(self):
        for expert_name in self.results.weights_history[0].keys():
            plt.plot(
                [loss[expert_name] for loss in self.results.losses_history],
                label=f"Loss of {expert_name}"
            )

        plt.plot(self.results.system_loss_history, label="System Loss", linestyle="--", color="black")

        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title("Expert and System Loss Over Time")
        plt.legend()
        plt.show()

    def plot_predictions(self):
        expert_names = list(self.results.prediction_history[0].keys())
        total_experts = len(expert_names)
        outcomes = len(next(iter(self.results.prediction_history[0].values())))  
        num_subplots = 4

        cumulative_predictions = {expert: np.zeros(outcomes) for expert in expert_names}
        for round_predictions in self.results.prediction_history:
            for expert, predictions in round_predictions.items():
                cumulative_predictions[expert] += predictions

        rows = 2 
        cols = 2 

        for start_expert in range(0, total_experts, num_subplots):
            fig, axes = plt.subplots(rows, cols)
            axes = axes.flatten() if num_subplots > 1 else [axes]

            for i, expert in enumerate(expert_names[start_expert:start_expert + num_subplots]):
                ax = axes[i]
                ax.bar(range(1, outcomes + 1), cumulative_predictions[expert])
                ax.set_title(f"Predictions of {expert}")
                ax.set_xlabel("Outcome")
                ax.set_ylabel("Cumulative Probability")

            plt.tight_layout()
            plt.show()


    def plot_theoretical_minimal_regret(self):
        total_experts = len(self.results.prediction_history[0].keys())
        optimal_regret = np.log(total_experts)

        plt.axhline(optimal_regret, color="red", linestyle="--", label=f"Optimal Min Loss (LogK={np.log(total_experts):.3f})")

    def plot_regret(self):
        plt.plot(self.results.regret_history, label="Regret", linestyle="--", color="black")

        plt.xlabel("Rounds")
        plt.ylabel("Regret")
        plt.title("Regret Over Time")
        plt.legend()
        plt.show()

    def plot_user_predictions(self):
        total_outcomes = len(self.results.system_prediction_history[0])
        outcome_counts = np.zeros(total_outcomes) 

        for predictions in self.results.system_prediction_history:
            outcome_counts += predictions

        outcome_counts /= np.sum(outcome_counts)

        plt.bar(range(1, total_outcomes + 1), outcome_counts)
        plt.xlabel("Predicted Outcome")
        plt.ylabel("Cumulative Probability")
        plt.title("System's cumulative Predictions")
        plt.show()
