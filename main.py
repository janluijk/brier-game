from experts import Expert
from update_weights import WeightUpdater
from visualize import Results, Visualizer
from config import CONFIG


def run_game(num_rounds=10):
    expert_config = CONFIG["experts"]

    experts = []

    for e in expert_config:
        experts.append(Expert(e["name"], e["description"], e["accuracy"]))

    num_rounds = CONFIG["num_rounds"]

    weight_updater = WeightUpdater(experts)
    results = Results()

    for _ in range(num_rounds):
        expert_predictions = {e.name: e.predict() for e in experts}
        true_outcome = 1

        scores = {name: (pred - true_outcome) ** 2 for name, pred in expert_predictions.items()}

        weight_updater.update(scores)
        final_prediction = weight_updater.get_optimal_predictions(scores)

        results.log(weight_updater.weights, final_prediction, scores)

    visualizer = Visualizer(results)

    visualizer.plot_weights()
    visualizer.plot_predictions()
    visualizer.plot_brier_scores()

if __name__ == "__main__":
    run_game()
