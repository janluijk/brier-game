from experts import RandomExpert, BiasedExpert
from weights import WeightManager 
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

    weight_manager = WeightManager(experts, total_outcomes)
    experiment_results = Results()
    prediction_optimizer = PredictionOptimizer(weight_manager)

    for _ in range(total_rounds):
        expert_forecast = {expert.name: expert.predict() for expert in experts}
        system_prediction = prediction_optimizer.get_optimal_predictions(expert_forecast)

        weight_manager.update_weights(expert_forecast)

        scores = {"name": 2}
        experiment_results.log(weight_manager.weights, system_prediction, scores)

    visualizer = Visualizer(experiment_results)
    visualizer.plot_weights()

if __name__ == "__main__":
    run_experiment()
