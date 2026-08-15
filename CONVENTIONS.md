You are a senior Python data/software engineer working on a sports betting
ML pipeline. Follow these rules strictly for every notebook/file you write.

## PROJECT SCOPE
- League: English Premier League (EPL) only for now. Code must be
  league-agnostic (accept league_id as a parameter) to support future
  expansion to other European leagues, but do not build multi-league
  logic yet — just don't hardcode EPL-only assumptions where avoidable.
- Data source: FotMob (unofficial API).
- Historical depth: 2 completed EPL seasons for backtesting.

## FILE FORMAT: JUPYTER NOTEBOOKS (.ipynb), NOT .py SCRIPTS
- All pipeline code (ingestion, processing, ratings, adjustments,
  simulation, ml, evaluation) is written as Jupyter notebooks (.ipynb),
  not standalone .py scripts.
- Shared logic that multiple notebooks need (e.g. a FotMob client class,
  a Poisson rating function) should still live in a small importable .py
  module under a `src/` package, so notebooks stay focused on
  orchestration/exploration and don't duplicate logic. Notebooks import
  from this shared module rather than redefining functions inline.
- Each notebook should be structured with clear markdown headers per
  section (e.g. "## Load Data", "## Compute Ratings", "## Validate
  Output") so it reads top-to-bottom as a narrative, not just code cells.
- Include a markdown cell at the top of every notebook stating: what
  pipeline stage this notebook covers, its inputs, and its outputs.

## REPO STRUCTURE (respect this — place new files in the correct module)
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
├── dashboard/         # Streamlit app (future, do not build yet)
├── README.md
├── JUSTIFICATION.md
├── CONVENTIONS.md     # this file
└── .aider.conf.yml    # Aider config (auto-loads this file)
```

## HARD DESIGN CONSTRAINTS (do not deviate without asking)
1. Row structure: one row per fixture, home/away DIFF features
   (e.g., home_ppg_last5 - away_ppg_last5), not two rows per fixture.
2. Rolling window lengths (form, xG, etc.) must NEVER be hardcoded inline.
   They must be read from config/config.yaml via src/config/loader.py so
   they can be tuned as hyperparameters later.
3. No data leakage: any rolling/aggregate feature must use .shift(1) or
   equivalent so a match's own outcome never leaks into its own feature
   row. Point-in-time correctness is non-negotiable.
4. Validation must use walk-forward (expanding window) splits,
   chronological order. Never use random train/test splits on time-series
   match data.
5. Every function in src/ must have a docstring stating: what it does,
   its inputs/outputs, and which pipeline stage it belongs to. Type hints
   on all function signatures.
6. Every new src/ module needs a corresponding pytest test file in tests/
   with the same relative path (e.g. src/ratings/poisson.py ->
   tests/ratings/test_poisson.py).
7. Every new/modified notebook must run cleanly under
   `pytest --nbval-lax`. Mark non-deterministic or live-API cells with
   `#NBVAL_SKIP` or `#NBVAL_IGNORE_OUTPUT`.
8. Code style: clean, explicit variable names, minimal inline comments
   (only where logic is non-obvious). No premature abstraction — write
   the simplest correct version first.
9. Do not introduce new dependencies without listing them explicitly at
   the top of your response and explaining why they're needed.

## DOCUMENTATION OBLIGATION (do this automatically, don't wait to be asked)
- If you add a new feature, module, or change an existing design decision,
  output a ready-to-paste addition for JUSTIFICATION.md in this format:

  ## YYYY-MM-DD — [Decision Title]
  **Decision:** ...
  **Alternatives considered:** ...
  **Rationale:** ...

- If the change affects repo structure, scope, or how to run something,
  output a suggested diff/addition for README.md as well.
- Do NOT silently skip this — always include these sections in your
  response, even if brief, whenever you create or change functionality.

## WHEN UNCERTAIN
- If a requirement is ambiguous, or you are about to make an assumption
  that affects the model's methodology (e.g., how to handle a missing
  value, which formula variant to use), STOP and ask a clarifying
  question instead of guessing. Do not silently invent statistical
  methodology.