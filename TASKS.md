# Task Backlog — Sports Betting ML Pipeline

All tasks follow the notebook + src/ split convention. Each task produces:
- A notebook in `notebooks/<stage>/`
- A tested module in `src/<stage>/` (if reusable logic is needed)
- Corresponding tests in `tests/`
- A JUSTIFICATION.md entry if design decisions are made

Priority: P0 (foundational) → P1 (core pipeline) → P2 (ML layer) → P3 (evaluation & polish)

---

## Phase 1: Foundation (Data + Ratings)

### P0 — Task 1.1: Data Ingestion via soccerdata (FBref + ClubElo)
**Notebook:** `notebooks/01_ingestion/01_soccerdata_ingest.ipynb`
**Module:** `src/ingestion/soccerdata_client.py`
**Tests:** `tests/ingestion/test_soccerdata_client.py`

**Goal:** Pull EPL fixtures, match stats, and Elo ratings for 2 historical seasons
using the maintained `soccerdata` library instead of a custom website scraper.

**Background:** The previous unofficial provider proved unstable and
undocumented in practice. `soccerdata` wraps FBref (fixtures, results, xG, shots,
player stats) and ClubElo (pre-computed team Elo ratings) into unified,
cached pandas DataFrames. See JUSTIFICATION.md entry for full rationale.

**Requirements:**
- `SoccerDataClient` class (thin wrapper around `soccerdata.FBref` and
  `soccerdata.ClubElo`) with methods:
  - `fetch_schedule(league: str, seasons: list[str]) -> pd.DataFrame`
    — wraps `FBref.read_schedule()`
  - `fetch_team_match_stats(league: str, seasons: list[str], stat_type: str) -> pd.DataFrame`
    — wraps `FBref.read_team_match_stats()` for goals, shots, xG
  - `fetch_player_season_stats(league: str, seasons: list[str], stat_type: str) -> pd.DataFrame`
    — wraps `FBref.read_player_season_stats()` (for future injury/absence work)
  - `fetch_elo_history(team: str | None = None) -> pd.DataFrame`
    — wraps `ClubElo.read_team_history()` or `ClubElo.read_by_date()`
- Do NOT reimplement caching/rate-limiting — `soccerdata` handles this
  internally (local cache under `~/soccerdata/data/`). Do not disable
  caching (`no_cache=True`) except in explicit test/debug scenarios.
- League and seasons must be parameters, not hardcoded, even though we
  start with `league="ENG-Premier League"` and 2 seasons.
- Save pulled data to Parquet under `data/raw/{season}/`:
  `schedule.parquet`, `team_match_stats.parquet`, `elo_history.parquet`.
- Notebook displays sample data and row counts; live scrape cells marked
  `#NBVAL_SKIP` (soccerdata's own cache makes repeated calls cheap, but
  first-run network calls should still be skipped in CI).

**Acceptance criteria:**
- [ ] Module has type hints, docstrings, and pytest tests using mocked/
      cached `soccerdata` responses (do not hit live network in tests).
- [ ] Notebook runs without error (skipping live-scrape cell in tests).
- [ ] No hardcoded league name or season strings inside functions —
      passed as parameters, with EPL/2-season as the default call in the notebook.
- [ ] Elo data successfully joins to fixtures on team name/date (flag any
      team-name mismatches between FBref and ClubElo naming conventions).
- [ ] JUSTIFICATION entry already added for the provider → soccerdata switch;
      add a follow-up entry only if team-name reconciliation logic is non-trivial.

---

### P0 — Task 1.2: Data Cleaning & Base Features
**Notebook:** `notebooks/02_processing/01_clean_features.ipynb`  
**Module:** `src/processing/clean_features.py`  
**Tests:** `tests/processing/test_clean_features.py`

**Goal:** Standardize IDs, handle missing values, derive base features (goals, shots, xG).

**Requirements:**
- Functions to:
  - Standardize team/player IDs across seasons and across FBref/ClubElo
    naming conventions (team-name reconciliation carried over from Task 1.1).
  - Impute or flag missing xG/shots values.
  - Compute base per-match features: goals for/against, shots for/against, xG for/against.
- Output: `data/processed/clean_fixtures.parquet`.

**Acceptance criteria:**
- [ ] Module tested for ID standardization, missing-value handling.
- [ ] Notebook shows missing-value rates before/after, sample cleaned data.
- [ ] No data leakage (cleaning uses only past data where applicable).
- [ ] JUSTIFICATION entry for imputation strategy.

---

### P0 — Task 1.3: Poisson Attack/Defense Ratings
**Notebook:** `notebooks/03_ratings/01_poisson_ratings.ipynb`  
**Module:** `src/ratings/poisson.py`  
**Tests:** `tests/ratings/test_poisson.py`

**Goal:** Compute rolling attack/defense strength per team using Poisson GLM,
with ClubElo ratings retained as a secondary cross-validation signal.

**Requirements:**
- `fit_poisson_ratings(fixtures: pd.DataFrame, home_field_advantage: bool = True)` returning attack/defense DataFrames.
- `compute_rolling_ratings(fixtures: pd.DataFrame, window: int)` using `.shift(1)` to avoid leakage.
- Read window from `src/config/config.yaml` via loader.
- Plot attack/defense strength over time for example teams, overlaid with
  ClubElo rating trend for sanity-checking.
- Save rolling ratings to `data/processed/ratings.parquet`.

**Acceptance criteria:**
- [ ] Module tested on synthetic data, verifies shift(1) usage.
- [ ] Notebook produces plots, saves ratings, uses config window.
- [ ] No hardcoded window length.
- [ ] JUSTIFICATION entry for statsmodels GLM choice.

---

## Phase 2: Domain Adjustments (Injuries)

### P1 — Task 2.1: Absence Classification
**Notebook:** `notebooks/04_adjustments/01_absence_classification.ipynb`  
**Module:** `src/adjustments/absence_classifier.py`  
**Tests:** `tests/adjustments/test_absence_classifier.py`

**Goal:** Tag each player-match as injury/suspension/rotation/cup-absence.

**Requirements:**
- Function to classify absence type based on player availability data
  (sourced from FBref player match logs via soccerdata — check minutes
  played per gameweek to infer absence; exact injury/suspension reason
  may require a supplementary source if FBref doesn't expose it directly).
- Output: `data/processed/absences.parquet` with columns: fixture_id, player_id, absence_type, position.

**Acceptance criteria:**
- [ ] Module tested with synthetic absence records.
- [ ] Notebook shows distribution of absence types per season.
- [ ] JUSTIFICATION entry for classification rules and data source used.

---

### P1 — Task 2.2: λ Adjustment Layer
**Notebook:** `notebooks/04_adjustments/02_lambda_adjustment.ipynb`  
**Module:** `src/adjustments/lambda_adjustment.py`  
**Tests:** `tests/adjustments/test_lambda_adjustment.py`

**Goal:** Adjust expected goals (λ) for home/away teams based on absences.

**Requirements:**
- Apply positional weights and decay factors from `src/config/config.yaml`.
- Function `adjust_lambda(base_lambda: float, absences: pd.DataFrame, team_id: str) -> float`.
- Output: `data/processed/adjusted_lambdas.parquet`.

**Acceptance criteria:**
- [ ] Module tested with known absence scenarios.
- [ ] Notebook shows before/after λ comparison for sample fixtures.
- [ ] Config-driven weights, no hardcoded numbers.
- [ ] JUSTIFICATION entry for weight values.

---

## Phase 3: Simulation

### P1 — Task 3.1: Poisson Scoreline Matrix
**Notebook:** `notebooks/05_simulation/01_scoreline_matrix.ipynb`  
**Module:** `src/simulation/scoreline_matrix.py`  
**Tests:** `tests/simulation/test_scoreline_matrix.py`

**Goal:** Convert adjusted λ into full scoreline probability grids.

**Requirements:**
- Function `build_scoreline_matrix(lambda_home: float, lambda_away: float, max_goals: int = 6) -> np.ndarray`.
- Output: scoreline probability matrix (7x7 grid for 0-6 goals each).

**Acceptance criteria:**
- [ ] Module tested against known Poisson calculations.
- [ ] Notebook visualizes heatmap for sample fixtures.
- [ ] JUSTIFICATION entry for max_goals choice.

---

### P1 — Task 3.2: Monte Carlo Simulation
**Notebook:** `notebooks/05_simulation/02_monte_carlo.ipynb`  
**Module:** `src/simulation/monte_carlo.py`  
**Tests:** `tests/simulation/test_monte_carlo.py`

**Goal:** Run 10,000+ trials per fixture, aggregate to W/D/L probabilities.

**Requirements:**
- Function `simulate_fixture(scoreline_matrix: np.ndarray, num_trials: int) -> dict` returning win/draw/loss probs.
- Read `num_trials` from `src/config/config.yaml`.
- Output: `data/processed/mc_probabilities.parquet`.

**Acceptance criteria:**
- [ ] Module tested for convergence (100 vs 1k vs 10k trials).
- [ ] Notebook shows convergence plot, saves probabilities.
- [ ] Config-driven trial count.
- [ ] JUSTIFICATION entry for trial count rationale.

---

## Phase 4: ML Layer

### P2 — Task 4.1: Feature Table Construction
**Notebook:** `notebooks/06_ml/01_feature_table.ipynb`  
**Module:** `src/ml/feature_table.py`  
**Tests:** `tests/ml/test_feature_table.py`

**Goal:** Build the fixture-level feature table for XGBoost training.

**Requirements:**
- Rolling form (5-6 match window), xG diff (6-10 match window), rating diffs
  (Poisson AND Elo differential), adjusted λ, MC probabilities, rest days, H2H.
- All windows from `src/config/config.yaml`.
- Strict `.shift(1)` usage to avoid leakage.
- Output: `data/processed/feature_table.parquet`.

**Acceptance criteria:**
- [ ] Module tested for leakage (verify shift(1) on rolling features).
- [ ] Notebook shows feature correlations, missing-value rates.
- [ ] Config-driven windows.
- [ ] JUSTIFICATION entry for feature selection.

---

### P2 — Task 4.2: XGBoost Model
**Notebook:** `notebooks/06_ml/02_xgboost_model.ipynb`  
**Module:** `src/ml/xgboost_model.py`  
**Tests:** `tests/ml/test_xgboost_model.py`

**Goal:** Train XGBoost classifier with walk-forward validation.

**Requirements:**
- Function `train_xgboost(features: pd.DataFrame, target: pd.Series, config: dict)` with walk-forward splits.
- Hyperparameters from `src/config/config.yaml`.
- Output: trained model artifact, feature importance report.

**Acceptance criteria:**
- [ ] Module tested with synthetic data, verifies walk-forward logic.
- [ ] Notebook shows feature importance, validation metrics.
- [ ] Config-driven hyperparameters.
- [ ] JUSTIFICATION entry for XGBoost over alternatives.

---

### P2 — Task 4.3: Ensemble Blend
**Notebook:** `notebooks/06_ml/03_ensemble_blend.ipynb`  
**Module:** `src/ml/ensemble_blend.py`  
**Tests:** `tests/ml/test_ensemble_blend.py`

**Goal:** Combine Poisson/MC probabilities with XGBoost probabilities.

**Requirements:**
- Function `blend_probabilities(poisson_probs: dict, xgboost_probs: dict, weights: dict) -> dict`.
- Weights from `src/config/config.yaml`.
- Output: `data/processed/ensemble_probabilities.parquet`.

**Acceptance criteria:**
- [ ] Module tested for probability sum = 1.
- [ ] Notebook compares blended vs. Poisson-only Brier scores.
- [ ] Config-driven weights.
- [ ] JUSTIFICATION entry for blend ratio.

---

## Phase 5: Evaluation & Feedback

### P2 — Task 5.1: Calibration Evaluation
**Notebook:** `notebooks/07_evaluation/01_calibration.ipynb`  
**Module:** `src/evaluation/calibration.py`  
**Tests:** `tests/evaluation/test_calibration.py`

**Goal:** Compute Brier score, log loss, reliability diagrams.

**Requirements:**
- Functions for Brier score, log loss, reliability diagram data.
- Output: `data/processed/calibration_metrics.parquet`, reliability diagram plot.

**Acceptance criteria:**
- [ ] Module tested against known Brier score examples.
- [ ] Notebook shows reliability diagram, Brier score trend over time.
- [ ] JUSTIFICATION entry for metric choices.

---

### P2 — Task 5.2: Recalibration
**Notebook:** `notebooks/07_evaluation/02_recalibration.ipynb`  
**Module:** `src/evaluation/recalibration.py`  
**Tests:** `tests/evaluation/test_recalibration.py`

**Goal:** Apply Platt scaling / isotonic regression to correct probabilities.

**Requirements:**
- Functions for Platt scaling and isotonic regression using sklearn.
- Output: `data/processed/recalibrated_probabilities.parquet`.

**Acceptance criteria:**
- [ ] Module tested on synthetic miscalibrated data.
- [ ] Notebook compares before/after Brier scores.
- [ ] JUSTIFICATION entry for recalibration method choice.

---

### P3 — Task 5.3: Dashboard (Future)
**Notebook:** N/A (Streamlit app)  
**Module:** `dashboard/app.py`  
**Tests:** `tests/dashboard/test_app.py`

**Goal:** Build a Streamlit dashboard showing live ratings, predictions, calibration.

**Requirements:**
- Display current team ratings, upcoming gameweek predictions, Brier score trend.
- Pull from `data/processed/` Parquet files.

**Acceptance criteria:**
- [ ] App runs locally, displays all three components.
- [ ] Basic tests for app rendering.
- [ ] JUSTIFICATION entry for dashboard design.

---

## Summary by Priority

| Priority | Tasks |
|----------|-------|
| P0 | 1.1 (Ingestion via soccerdata), 1.2 (Cleaning), 1.3 (Poisson ratings) |
| P1 | 2.1 (Absence classification), 2.2 (λ adjustment), 3.1 (Scoreline matrix), 3.2 (Monte Carlo) |
| P2 | 4.1 (Feature table), 4.2 (XGBoost), 4.3 (Ensemble), 5.1 (Calibration), 5.2 (Recalibration) |
| P3 | 5.3 (Dashboard) |

Start with P0 tasks in order — each depends on the previous one's output.