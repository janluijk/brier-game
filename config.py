CONFIG = {
    # Experts configuration

    "experts": [
        {"name": "1st", "description": "Chooses randomly", "accuracy": 0.9},
        {"name": "2nd", "description": "Chooses randomly", "accuracy": 0.1},
    ],

    # Outcome space
    "omega_size": 1,

    "num_rounds": 1000,

    "visualisation": {
        "plot_weights": True,
        "plot_predictions": True,
        "plot_brier_scores": True,
    }
}
