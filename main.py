from experts import BiasedExpert
from expert_performance import ExpertPerformanceTracker
from predictions import PredictionOptimizer
from visualize import Results, Visualizer
from config import CONFIG

def apply_probability_shifts(expert, round_num):
    for shift in CONFIG["probability_shifts"]:
        if shift["round"] == round_num and shift["expert"] == expert.name:
            expert.update_probabilities(shift["bias"])

def run_experiment():
    expert_settings       = CONFIG["experts"]
    total_rounds          = CONFIG["total_rounds"]
    total_outcomes        = CONFIG["total_outcomes"]
    
    experts = []
    for e in expert_settings:
        experts.append(BiasedExpert(e["name"], e["bias"], total_outcomes))

    expert_manager = ExpertPerformanceTracker(experts, total_outcomes)
    experiment_results = Results()
    prediction_optimizer = PredictionOptimizer(expert_manager)

    for round_num in range(total_rounds):

        for expert in experts:
            apply_probability_shifts(expert, round_num)

        expert_predictions = {expert.name: expert.predict() for expert in experts}
        system_prediction = prediction_optimizer.get_optimal_predictions(expert_predictions)

        expert_manager.update_weights(expert_predictions)
        expert_manager.update_expert_losses(expert_predictions)
        expert_manager.update_system_loss(system_prediction)

        experiment_results.log(expert_manager.weights, expert_predictions, expert_manager.losses, expert_manager.system_loss, system_prediction)

    visualizer = Visualizer(experiment_results)

    if CONFIG["visualization"]["plot_weights"]:
        visualizer.plot_weights()
    if CONFIG["visualization"]["plot_optimal_loss"]:
        visualizer.plot_theoretical_minimal_loss()
    if CONFIG["visualization"]["plot_losses"]:
        visualizer.plot_losses()
    if CONFIG["visualization"]["plot_predictions"]:
        visualizer.plot_predictions()
    if CONFIG["visualization"]["plot_user_predictions"]:
        visualizer.plot_user_predictions()

if __name__ == "__main__":
    run_experiment()
