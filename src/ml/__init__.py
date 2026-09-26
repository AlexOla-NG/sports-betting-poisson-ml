"""
Machine Learning package for sports betting pipeline.
"""

from src.ml.feature_table import (
    construct_feature_table,
    build_rolling_team_features,
    compute_head_to_head_features,
)

__all__ = [
    "construct_feature_table",
    "build_rolling_team_features",
    "compute_head_to_head_features",
]
