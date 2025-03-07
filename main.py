from experts import Expert, StrongExpert, WeakExpert
from update_weights import WeightUpdater
from visualize import Results, Visualizer

def brier_score(prediction, outcome):
    """Computes the Brier score: lower is better."""
    return (prediction - outcome) ** 2

def run_game(num_rounds=10):
    experts = [
        Expert("Random"),
        StrongExpert("Strong"),
        WeakExpert("Weak")
    ]
    
    results = Results()

    weight_updater = WeightUpdater(experts)

    for _ in range(num_rounds):
        expert_predictions = {e.name: e.predict() for e in experts}
        true_outcome = 1

        scores = {name: (pred - true_outcome) ** 2 for name, pred in expert_predictions.items()}

        weight_updater.update(scores)
        final_prediction = weight_updater.get_weighted_prediction(expert_predictions)

        results.log(weight_updater.weights, final_prediction, scores)

    visualizer = Visualizer(results)

    visualizer.plot_weights()
    visualizer.plot_predictions()
    visualizer.plot_brier_scores()

run_game(100)
