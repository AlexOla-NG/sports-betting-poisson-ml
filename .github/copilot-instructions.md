# .github/copilot-instructions.md — GitHub guidance for Copilot / agents

Purpose: short, GitHub-focused guidance for automated agents and Copilot when
working on pull requests in this repository. Agents should still prefer
`AGENTS.md` for full conventions — this file highlights repository-specific
hints for PRs and CI.

- Primary reference: [AGENTS.md](../AGENTS.md) and [CONVENTIONS.md](../CONVENTIONS.md)
- Tests: run `pytest -q` for unit tests and `pytest --nbval-lax` for notebooks.
- Config: use [src/config/config.yaml](../src/config/config.yaml) as the canonical hyperparameter file.
- When opening a PR, ensure:
  - New `src/` modules include docstrings, type hints, and a matching test in [tests/](../tests/).
  - Rolling-window hyperparameters are read from the config file rather than hardcoded.
  - Any notebook changes validate under `pytest --nbval-lax` or mark live-API cells with `#NBVAL_SKIP`.
  - Required imports are grouped at the top of each module and notebook setup cell: standard library, third-party packages, then local `src` imports. Avoid scattered or conditional imports.
  - Add or update the design log in [JUSTIFICATION.md](../JUSTIFICATION.md); if tooling requires a compatibility pointer, keep it brief in same file.

Do not:
- Commit large data under `data/` (it's gitignored).
- Commit test results.
- Commit virtual environment data.
- Commit pycache files and folders
- Commit live API credentials or secrets.

If unsure about a design choice that affects modeling or data leakage, stop
and ask a reviewer rather than guessing.

---

## Common PR Patterns & Checks

When opening a PR with a new notebook or src/ module:

### Notebooks
- [ ] Top markdown cell documents: stage name, what inputs are expected, what outputs are produced
- [ ] Run `pytest --nbval-lax` to ensure notebook executes without error
- [ ] Mark live data-source cells with `#NBVAL_SKIP` (e.g., soccerdata FBref calls)
- [ ] Mark output-changing cells with `#NBVAL_IGNORE_OUTPUT` (e.g., row counts, timestamps)
- [ ] Sections are clearly labeled with markdown headers (`## Load`, `## Transform`, etc.)
- [ ] No hardcoded hyperparameters (window lengths, thresholds) — read from src/config/config.yaml
- [ ] Required imports are centralized in the first executable setup cell and grouped by standard library, third-party, and local imports

### src/ Modules
- [ ] Every function has: docstring (purpose, inputs, outputs), type hints
- [ ] Every module states which pipeline stage it belongs to (e.g., "Rating stage")
- [ ] Matching test file exists in tests/ with same relative path
  - Example: `src/ratings/poisson.py` → `tests/ratings/test_poisson.py`
- [ ] Tests use mocked external calls (no live API, no file I/O to data/)
- [ ] New dependency added? List it explicitly in PR description with justification

### JUSTIFICATION.md & README
- [ ] If you added a feature or changed design, add an entry to JUSTIFICATION.md:
  ```
  ## YYYY-MM-DD — [Descriptive Title]
  **Decision:** What was decided
  **Alternatives considered:** What else was considered
  **Rationale:** Why this was chosen
  ```
- [ ] If scope/structure changed, update README.md repo diagram

### Config & Rolling Windows (Critical)
- [ ] No hardcoded window lengths, weights, or hyperparameters in code
- [ ] All tunable values live in `src/config/config.yaml`
- [ ] Load via:
  ```python
  from src.config.loader import load_config, get_rating_window, get_form_windows
  config = load_config()
  rating_window = get_rating_window(config)
  ```

### Data Leakage Prevention (Critical for ML)
- [ ] All rolling/aggregate features use `.shift(1)` to ensure point-in-time correctness
- [ ] One row per fixture (home/away DIFF features), never two rows per fixture
- [ ] Walk-forward validation used (never random train/test splits on time-series data)
- [ ] Feature engineering logic is deterministic (no leakage from test set into training features)
