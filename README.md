 This Python program simulates the Brier game, modelling expert forecasting with adaptive exponential weighting using the Vovk-Zhdanov algorithm.

## Requirements
- Python 3.x
- Git

## Installation
Clone the repo: `git clone https://github.com/janluijk/brier-game.git`
Install the required libraries: `pip install -r requirements.txt`

## Running the program
`python main.py`

## Configuring the program
Edit `config.py` as needed. Options include:
- `experts`: List of experts and their biases.
- `probability_shifts`: List of changes of expert biases during specific rounds.
- `total_outcomes`: Total possible outcomes of a round.
- `total_rounds`: Total number of rounds for the simulation.
- `visualization`: Toggle different visualizations for the simulation

The `bias` value assigned to each expert represents the probability that the selected outcome is the true outcome. The remaining probability, `1 - bias`, represents the likelihood that the expert's selected outcome is incorrect. This is implemented as a uniform distribution across all other possible outcomes.

