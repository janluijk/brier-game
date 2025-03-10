from experts import RandomExpert, BiasedExpert
from expert_performance import ExpertPerformanceTracker
from predictions import PredictionOptimizer
from visualize import Results, Visualizer
from config import CONFIG

def run_experiment():
    expert_settings       = CONFIG["experts"]
    total_rounds          = CONFIG["total_rounds"]
    total_outcomes        = CONFIG["total_outcomes"]
    
    experts = []
    for e in expert_settings:
        if e.get("bias") is not None:
            experts.append(BiasedExpert(e["name"], e["bias"], total_outcomes))
        else:
            experts.append(RandomExpert(e["name"], total_outcomes))

    expert_manager = ExpertPerformanceTracker(experts, total_outcomes)
    experiment_results = Results()
    prediction_optimizer = PredictionOptimizer(expert_manager)

    for _ in range(total_rounds):
        expert_predictions = {expert.name: expert.predict() for expert in experts}
        system_prediction = prediction_optimizer.get_optimal_predictions(expert_predictions)

        expert_manager.update_weights(expert_predictions)
        expert_manager.update_expert_losses(expert_predictions)
        expert_manager.update_system_loss(system_prediction)

        experiment_results.log(expert_manager.weights, expert_predictions, expert_manager.losses, expert_manager.system_loss)
        # weights, loss, system_loss, expert_predictions, system_prediction, 

    visualizer = Visualizer(experiment_results)

    if CONFIG["visualization"]["plot_weights"]:
        visualizer.plot_weights()
    if CONFIG["visualization"]["plot_optimal_loss"]:
        visualizer.plot_theoretical_minimal_loss()
    if CONFIG["visualization"]["plot_losses"]:
        visualizer.plot_losses()
    if CONFIG["visualization"]["plot_predictions"]:
        visualizer.plot_predictions()

if __name__ == "__main__":
    run_experiment()
