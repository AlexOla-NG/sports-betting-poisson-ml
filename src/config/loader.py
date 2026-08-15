"""
Central configuration loader for the sports betting pipeline.
All hyperparameters and tuning knobs are read from config/config.yaml.
Never hardcode window lengths, weights, or other magic numbers in code.
"""

from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the central configuration file.

    Behavior:
    - If `config_path` is provided, try that path relative to repo root.
    - Otherwise, prefer `src/config/config.yaml`, then fall back to
      `config/config.yaml` for backward compatibility.

    Parameters
    ----------
    config_path : str | None
        Optional path to the YAML config file, relative to repo root.

    Returns
    -------
    dict[str, Any]
        Nested dictionary of configuration values.
    """
    repo_root = Path(__file__).resolve().parents[2]

    candidates = []
    if config_path:
        candidates.append(repo_root / config_path)
    else:
        candidates.append(repo_root / "src/config/config.yaml")
        candidates.append(repo_root / "config/config.yaml")

    tried = []
    for path in candidates:
        tried.append(str(path))
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        "Config file not found. Tried: " + ", ".join(tried)
    )


# Convenience accessors for commonly-used values

def get_rating_window(config: dict[str, Any]) -> int:
    """Return the rolling window for Poisson attack/defense ratings."""
    return config["ratings"]["rating_window"]


def get_form_windows(config: dict[str, Any]) -> tuple[int, int]:
    """Return (points_window, goals_window) for form features."""
    return config["form"]["points_window"], config["form"]["goals_window"]


def get_xg_windows(config: dict[str, Any]) -> tuple[int, int]:
    """Return (xg_for_window, xg_against_window) for xG features."""
    return config["xg"]["xg_for_window"], config["xg"]["xg_against_window"]


def get_mc_trials(config: dict[str, Any]) -> int:
    """Return number of Monte Carlo trials per fixture."""
    return config["monte_carlo"]["num_trials"]