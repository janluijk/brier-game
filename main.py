from experts import RandomExpert, BiasedExpert
from expert_performance import ExpertPerformanceTracker
from predictions import PredictionOptimizer
from visualize import Results, Visualizer
from config import CONFIG

def run_experiment():
    expert_settings       = CONFIG["experts"]
    total_rounds          = CONFIG["num_rounds"]
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

        scores = {"name": 2}
        experiment_results.log(expert_manager.weights, system_prediction, scores)

    visualizer = Visualizer(experiment_results)
    visualizer.plot_weights()

if __name__ == "__main__":
    run_experiment()
