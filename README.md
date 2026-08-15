# Sports Betting ML Pipeline

Football match prediction pipeline for EPL, built on Poisson attack/defense
ratings, injury-adjusted expected goals, Monte Carlo simulation, and an
XGBoost ensemble correction layer.

## Scope (current)
- **League:** English Premier League (EPL) only
- **Seasons:** 2 most recent completed seasons (backtesting depth)
- **Data source:** FotMob (unofficial API)
- **Future:** Expand to all major European leagues once EPL pipeline is validated

## Repo Structure
```
sports-betting/
├── src/               # Shared importable Python modules (thin, tested)
│   ├── ingestion/
│   ├── processing/
│   ├── ratings/
│   ├── adjustments/
│   ├── simulation/
│   ├── ml/
│   ├── evaluation/
│   └── config/         # Configuration loader (reads config/config.yaml)
├── notebooks/         # .ipynb notebooks, one per pipeline stage/task
│   ├── 01_ingestion/
│   ├── 02_processing/
│   ├── 03_ratings/
│   ├── 04_adjustments/
│   ├── 05_simulation/
│   ├── 06_ml/
│   └── 07_evaluation/
├── tests/             # pytest unit tests for src/, mirroring structure
├── config/            # Tunable hyperparameters (YAML)
├── data/              # raw/ and processed/ data, gitignored
├── dashboard/         # Streamlit app (future)
├── README.md          # This file
├── JUSTIFICATION.md   # Design decisions and rationale log
├── CONVENTIONS.md     # Persistent AI context (loaded by Aider)
└── .aider.conf.yml    # Aider config (auto-loads CONVENTIONS.md)
```

## Design Decisions
- **Row structure:** One row per fixture, home/away diff features (not two rows per fixture).
- **Rolling windows:** Treated as tunable hyperparameters via `config/config.yaml`, not hardcoded.
  Starting priors: form 5-6 matches, xG 6-10 matches, ratings 8 matches — to be tuned empirically.
- **Validation:** Walk-forward (expanding window), never random train/test split.
- **Model:** Poisson/Monte Carlo baseline + XGBoost correction layer, blended ensemble.
- **Evaluation metric:** Brier score + reliability diagrams (target < 0.20).
- **File format:** Jupyter notebooks (.ipynb) for pipeline stages, with shared logic in `src/` modules.

## Development Environment
- **Editor:** VS Code
- **AI tooling:** Aider (`aider --model ollama/qwen2.5-coder:7b`) with persistent context via `CONVENTIONS.md`
- **Orchestration:** Manual notebook/script runs (no scheduler yet)
- **Testing:** `pytest` for src/ modules, `pytest --nbval-lax` for notebooks

## Testing
```bash
pytest tests/
pytest --nbval-lax notebooks/
```
Notebooks with non-deterministic or live-API cells should use `#NBVAL_SKIP` markers.

## Configuration
All hyperparameters live in `config/config.yaml` — never hardcode window lengths,
weights, or other magic numbers in code. Load via:

```python
from src.config.loader import load_config, get_rating_window, get_form_windows

config = load_config()
rating_window = get_rating_window(config)
points_window, goals_window = get_form_windows(config)
```

## Maintenance Rule
Every new feature or design change must be accompanied by:
1. An update to this README (if scope/structure changes)
2. An update to `JUSTIFICATION.md` (rationale, alternatives considered, sources)
3. A corresponding test (unit test or notebook test)

## Agent Guidance
- Agent and automated tool guidance: see [AGENTS.md](AGENTS.md) for concise, actionable instructions for AI coding agents and automated edits.
- GitHub/Copilot PR guidance: see [.github/copilot-instructions.md](.github/copilot-instructions.md).