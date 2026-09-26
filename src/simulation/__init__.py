"""
Simulation package for sports betting pipeline.
"""

from src.simulation.scoreline_matrix import (
    build_scoreline_matrix,
    extract_1x2_probabilities,
    extract_over_under_probabilities,
    extract_btts_probability,
    extract_most_likely_scoreline,
    compute_fixture_scoreline_outcomes,
)

from src.simulation.monte_carlo import (
    simulate_fixture_trials,
    simulate_fixture_from_matrix,
    run_monte_carlo_dataset,
)

__all__ = [
    "build_scoreline_matrix",
    "extract_1x2_probabilities",
    "extract_over_under_probabilities",
    "extract_btts_probability",
    "extract_most_likely_scoreline",
    "compute_fixture_scoreline_outcomes",
    "simulate_fixture_trials",
    "simulate_fixture_from_matrix",
    "run_monte_carlo_dataset",
]
