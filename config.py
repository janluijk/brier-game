CONFIG = {

    "experts": [
        # {"type": "RandomExpert", "name": "0th"},
        {"type": "BaisedExpert", "name": "1st", "bias": 0.1},
        {"type": "BaisedExpert", "name": "2nd", "bias": 0.2},
        {"type": "BaisedExpert", "name": "3rd", "bias": 0.3},
        {"type": "BaisedExpert", "name": "4th", "bias": 0.4},
        {"type": "BaisedExpert", "name": "5th", "bias": 0.5},
        {"type": "BaisedExpert", "name": "6th", "bias": 0.6},
        {"type": "BaisedExpert", "name": "7th", "bias": 0.7},
        {"type": "BaisedExpert", "name": "8th", "bias": 0.8},
        {"type": "BaisedExpert", "name": "9th", "bias": 0.9},
        {"type": "BaisedExpert", "name": "10th", "bias": 1.0},
    ],

    "total_outcomes": 5,

    "total_rounds": 10,

    "visualisation": {
        "plot_weights": True,
        "plot_predictions": False,
        "plot_brier_scores": False,
    }
}
