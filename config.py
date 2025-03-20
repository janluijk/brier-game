CONFIG = {

    "experts": [
        {"name": "Random", "bias": 0.2}, # a bias of (1.0 / total_outcomes) results in a random distribution

        {"name": "Expert1", "bias": 0.1},
        {"name": "Expert2", "bias": 0.3},
        {"name": "Expert3", "bias": 0.5},
        {"name": "Expert4", "bias": 0.7},
        {"name": "Expert5", "bias": 0.9},
    ],

    "probability_shifts": [
        { "round": 15, "expert": "Expert1", "bias": 0.6},
        { "round": 15, "expert": "Expert1", "bias": 0.6},
        { "round": 15, "expert": "Expert1", "bias": 0.6}
    ],

    "total_outcomes": 5,

    "total_rounds": 30,

    "visualization": {
        "plot_weights": True,
        "plot_losses": True,
        "plot_optimal_loss": True,
        "plot_predictions": True,
        "plot_user_predictions": True
    }
}
