# AGENTS.md — Guidance for AI coding agents

Purpose: provide concise, actionable instructions to AI coding agents working in this repository so they can be immediately productive and follow project conventions.

Quick links
- CONVENTIONS: [CONVENTIONS.md](CONVENTIONS.md)
- Config: [src/config/config.yaml](src/config/config.yaml)
- Notebooks: `notebooks/` (one notebook per pipeline stage)
- Shared code: `src/`
- GitHub/Copilot guidance: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- Agent check script: `scripts/agent_customize_checks.py` (run locally or in CI)
- CI workflow: [.github/workflows/agent-customization-check.yml](.github/workflows/agent-customization-check.yml)
- Justification log: `JUSTIFICATION.md`

What this project expects from AI agents
- Primary code artifacts are Jupyter notebooks under `notebooks/` (NOT .py scripts). Keep notebook structure narrative and include an inputs/outputs top cell per notebook.
- Shared logic belongs in `src/` (small, well-documented modules). Every new `src/` module requires a corresponding test under `tests/` mirroring the path.
- Follow the rules in [CONVENTIONS.md](CONVENTIONS.md) strictly (rolling windows from config, no data leakage, walk-forward validation, docstrings/type hints, nbval testing constraints).
- Keep imports at the top of each Python module and notebook setup cell, grouped as standard library, third-party dependencies, then local `src` imports. Do not add conditional or scattered imports inside processing cells/functions unless a dependency is genuinely optional and the exception is documented.

Test and validation commands (use these in CI and locally)
- Run unit tests: `pytest -q`
- Validate notebooks: `pytest --nbval-lax`
- Agent docs check (local): `python3 scripts/agent_customize_checks.py`

Agent behavior: do this
- Prefer editing or adding small `src/` modules rather than embedding large functions inline in notebooks.
- Place all required imports at the top of Python modules and in the first executable notebook cell so dependencies are visible before pipeline logic begins.
- When editing/adding files, include docstrings with inputs/outputs and pipeline stage, and add a pytest file under `tests/` for that module.
- For any change that affects repo structure or design decisions, add an entry to `src/config/JUSTIFICATION.md` (see [CONVENTIONS.md](CONVENTIONS.md)).
- When encountering ambiguous design choices, STOP and ask a clarifying question rather than guessing.
- Always update `README.md` and `CONVENTIONS.md` when scope or structure changes.

Agent behavior: do NOT do this
- Do not hardcode hyperparameters (window lengths, etc.) — read them from `src/config/config.yaml`.
- Do not commit large data files under `data/` — data is gitignored.
- Do not run or commit live API credentials. Mark live-API notebook cells with `#NBVAL_SKIP` or `#NBVAL_IGNORE_OUTPUT`.
- Do not modify `CONVENTIONS.md` without discussing with the maintainer; it's used as persistent AI context.

Where to look first
- Start with [CONVENTIONS.md](CONVENTIONS.md) and `src/config/config.yaml` for hyperparameter and pipeline rules.
- For tests and examples, inspect `tests/` and `notebooks/`.

If you plan to add new agent customizations (skills, prompts, hooks)
- Keep them minimal and link to existing docs instead of copying content.
- Name new files clearly (e.g., `AGENTS.md`, `.github/copilot-instructions.md`) and preserve existing content when updating.
- Add a short JUSTIFICATION.md entry describing the customization and rationale.

If you have questions or want to propose additional agent hooks, ask the maintainer and include a short justification entry for `src/config/JUSTIFICATION.md`.

---

## Quick Workflow: Adding a New Pipeline Stage

If you need to add a new stage (beyond the current 7):

1. **Create the notebook directory** (respecting 01-prefix ordering):
   ```bash
   mkdir -p notebooks/NN_stagename/
   ```

2. **Create the shared module** (if needed):
   ```bash
   mkdir -p src/stagename/
   touch src/stagename/__init__.py
   touch src/stagename/module.py
   ```

3. **Create a matching test file**:
   ```bash
   mkdir -p tests/stagename/
   touch tests/stagename/test_module.py
   ```

4. **Notebook structure** (see [01_soccerdata_ingest.ipynb](notebooks/01_ingestion/01_soccerdata_ingest.ipynb) as template):
   - Top markdown cell: stage name, inputs, outputs
   - Section headers: Load Data → Transform → Validate
   - Save to `data/processed/{stage_name}/` as Parquet
   - Mark live API / non-deterministic cells with `#NBVAL_SKIP`

5. **Update JUSTIFICATION.md**:
   ```
   ## YYYY-MM-DD — [Stage Name]
   **Decision:** Why this stage exists, what it does
   **Rationale:** Why this stage is necessary for the pipeline
   ```

6. **Update README.md** with new stage in repo structure diagram

---

## Quick Reference: Loading Config & Rolling Windows

Use this pattern in notebooks and src/ modules:

```python
from src.config.loader import load_config, get_rating_window, get_form_windows

config = load_config()

# Single parameter
rating_window = get_rating_window(config)  # e.g., 8

# Multiple parameters  
points_window, goals_window = get_form_windows(config)  # e.g., (5, 5)

# Direct access to full config dict
xg_for_window = config['xg']['xg_for_window']
```

Never hardcode these values. Always read from config.yaml.

---

## Common Pitfalls: Data Leakage & Row Structure

**❌ DO NOT do this:**

```python
# BAD: feature uses current match's own outcome
df['home_xg_trend'] = df['home_xg'].rolling(5).mean()  # No .shift()!

# BAD: two rows per fixture (doubles dataset, violates row structure)
df = df.melt(id_vars=['fixture_id', ...], var_name='team', ...)

# BAD: hardcoded window length
df['form'] = df.groupby('team')['points'].rolling(5).mean()
```

**✅ DO this:**

```python
# GOOD: .shift(1) ensures point-in-time correctness
df['home_xg_trend'] = df.sort_values('fixture_date').groupby('home_team')['home_xg'].transform(lambda x: x.rolling(5).mean().shift(1))

# GOOD: one row per fixture, home-away DIFF features
df['xg_diff'] = df['home_xg'] - df['away_xg']

# GOOD: load window from config, never hardcode
window = get_form_windows(config)[0]
df['form'] = df.groupby('team')['points'].transform(lambda x: x.rolling(window).mean().shift(1))
```

---

## Notebook Testing: `#NBVAL_SKIP` & Output Validation

- `#NBVAL_SKIP` cells are **skipped entirely** during `pytest --nbval-lax` runs
  - Use for: live API calls, manual exploration, non-deterministic randomness
  - Example:
    ```python
    # NBVAL_SKIP
    df = client.fetch_fixtures(...)  # live API call
    ```

- `#NBVAL_IGNORE_OUTPUT` cells **execute** but output is not validated
  - Use for: cells where output legitimately changes (e.g., new data added)
  - Example:
    ```python
    # NBVAL_IGNORE_OUTPUT
    ratings_df.to_parquet(...)  # row count changes each season
    print(f"Saved {len(ratings_df)} rows")
    ```

- No marker: output is validated to match saved notebook state
  - Use for: deterministic logic, assertions, static data checks

Run tests locally before pushing:
```bash
pytest --nbval-lax  # Validate all notebooks
pytest -q           # Unit tests only
```
