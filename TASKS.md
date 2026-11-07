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

**Corrected data-sourcing step:** Use Understat player-match appearances via
`soccerdata.Understat.read_player_match_stats()` with no `match_id`, join dates
from `read_schedule()`, and infer missing rows only for pre-fixture key starters
identified by the rolling minute-share rule. Do not use Transfermarkt or
cross-join all players against all fixtures.

**Requirements:**
- Function to classify absence type based on player availability data
  (sourced from Understat player match appearances via soccerdata — infer
  missing appearances only for established pre-fixture key starters; exact
  injury/suspension reason is intentionally not observed).
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
- [x] Module tested against known Poisson calculations.
- [x] Notebook visualizes heatmap for sample fixtures.
- [x] JUSTIFICATION entry for max_goals choice.

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
- [x] Module tested for convergence (100 vs 1k vs 10k trials).
- [x] Notebook shows convergence plot, saves probabilities.
- [x] Config-driven trial count.
- [x] JUSTIFICATION entry for trial count rationale.

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
- [x] Module tested for leakage (verify shift(1) on rolling features).
- [x] Notebook shows feature correlations, missing-value rates.
- [x] Config-driven windows.
- [x] JUSTIFICATION entry for feature selection.

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
- [x] Module tested with synthetic data, verifies walk-forward logic.
- [x] Notebook shows feature importance, validation metrics.
- [x] OOF permutation importance and fixed-hyperparameter ablation validation added.
- [x] Config-driven hyperparameters.
- [x] JUSTIFICATION entry for XGBoost over alternatives.

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
- [x] Module tested for probability sum = 1.
- [x] Notebook compares blended vs. Poisson-only Brier scores.
- [x] Config-driven weights.
- [x] JUSTIFICATION entry for blend ratio.

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
- [x] Module tested against known Brier score examples.
- [x] Notebook shows reliability diagram, Brier score trend over time.
- [x] JUSTIFICATION entry for metric choices.

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
- [x] Module tested on synthetic miscalibrated data.
- [x] Notebook compares before/after Brier scores.
- [x] JUSTIFICATION entry for recalibration method choice.

---

### P2 — Task 5.3: Recent-Match Evaluation & Upcoming-Fixture Prediction
**Status:** Planned — capability audit is the immediate next action.
**Proposed notebook:** `notebooks/08_prediction/01_recent_and_upcoming.ipynb`  
**Proposed module:** `src/prediction/predict_fixtures.py`  
**Proposed tests:** `tests/prediction/test_predict_fixtures.py`

Paths are proposed, not claims that these files already exist. Adapt to repository
conventions and reuse existing ingestion, features, ratings, simulation, and
model functions. Keep reusable logic in tested modules and orchestration in
the notebook. Do not create a parallel modeling pipeline.

**Goal:** Evaluate the frozen selected specification on newer completed matches,
then generate usable forecasts for upcoming fixtures. No dashboard is required.

#### 5.3a — Repository & Data Capability Audit (run first)

**Requirements:**
- Inspect repository conventions, config loader, ingestion interfaces, and saved artifacts.
- Report the latest raw/processed match dates and available seasons.
- Verify whether the current provider supports newer results and scheduled fixtures;
  do not assume live availability or introduce a replacement provider silently.
- Determine whether a fitted context-only XGBoost model exists, or only OOF predictions.
- Locate the exact selected feature list, order, preprocessing settings, and model parameters.
  Do not reconstruct the context-only schema by an ad hoc exclusion list.
- Check whether unplayed rows survive ingestion and feature generation with missing
  outcomes, goals, xG, and targets.
- Verify pre-match generation of ratings, expected goals, MC probabilities, and other inputs.
- Audit historical availability of absence inputs. A missing appearance in the match
  being predicted must not be treated as pre-match evidence of an absence.
- Identify promoted/new teams, insufficient histories, postponed matches, and team-ID mismatches.
- Report blockers and the smallest required implementation changes before proceeding.

**Acceptance criteria:**
- [x] Audit summary recorded in the notebook with actual schemas, dates, artifacts, and gaps.
- [x] Data source supports both completed and upcoming fixtures, or a blocker is documented.
- [x] Exact context-only feature schema and training specification identified.
- [x] Historical/as-of availability limitations are explicit; unresolved inputs are not silently fabricated.

#### 5.3b — Frozen-Model Evaluation on Recent Completed Matches

**FBref fallback status:** Implemented separately as `fbref_fallback_v1` for
offline newer-completed-match evaluation. The original xG-based specification
remains blocked by unavailable current-season xG and is not replaced.

**Requirements:**
- Initial proposed historical training cutoff: 2026-05-24 (end of reported development history).
  Confirm against actual data; store it in config rather than hardcoding function logic.
- Select an available completed-match period after the cutoff and record its dates
  before inspecting performance. Verify whether these fixtures are genuinely unused
  in model/feature/weight selection; otherwise label evaluation accordingly.
- Load a matching fitted model, or fit the fixed context-XGBoost specification using
  only eligible completed training fixtures through the cutoff. Save its feature schema,
  class mapping, config snapshot, training cutoff, and model identifier.
- Freeze XGBoost fitted parameters during this first recent-match evaluation.
- Generate predictions chronologically. Rolling state may update from earlier completed
  games, but no match may use its own result or future data.
- Keep same-date fixtures atomic if reliable kickoff/result-availability times are unavailable.
- Do not insert historical results into prediction inputs; attach evaluation targets only
  after forecasts are generated. Updating history after a completed block is allowed.
- Reconstruct pre-match inputs as of each prediction cutoff, including ratings and
  availability information. If historical inputs cannot be reconstructed, stop or
  explicitly document a proposed fallback for approval; do not silently change the model.
- Use the selected 80/20 blend and identity post-processing unchanged.
- Report fixture count, exclusions/reasons, dates, accuracy, multiclass summed Brier,
  and log loss; display fixture-level probabilities and actual outcomes.
- Do not tune features, blend weights, calibration, or model settings on this evaluation period.

**Proposed outputs:**
- `data/processed/recent_match_predictions.parquet`
- `data/processed/recent_match_metrics.parquet`
- A versioned fitted model and metadata in the repository's existing artifact location.

**Acceptance criteria for the FBref fallback only:**
- [x] Fixed training cutoff and evaluation scope recorded before scoring.
- [x] Model, feature schema, class mapping, and metadata saved and reloadable.
- [x] Predictions use only information available at each historical cutoff.
- [x] Fixture-level predictions and actuals join one-to-one; exclusions are explicit.
- [x] Saved predictions reproduce reported metrics.
- [x] Recent results are reported honestly, without retuning or implying guaranteed accuracy.

#### 5.3c — Upcoming-Fixture Inference

**Requirements:**
- After recording the recent evaluation, fit a separate current-model version on all
  eligible completed data through a declared current training cutoff, using the same
  selected specification. Preserve the historical evaluation model and outputs.
- Fetch scheduled fixtures for a configurable date range or upcoming gameweek.
- Use the same feature transformations and ordered schema as training.
- Keep unplayed fixture targets/results missing; do not drop them for lacking outcomes.
- Build features only from history available at the declared prediction/data cutoff.
- Do not allow one upcoming fixture to supply invented results or statistics to another.
- Validate team mapping and history; flag or stop on unsupported fixtures rather than
  silently substituting ratings, injuries, or features.
- Produce XGBoost and MC probabilities and the fixed 80/20 blended prediction.
- Display fixture, kickoff/date, home/draw/away probabilities, and highest-probability
  outcome. Do not invent an additional confidence score or betting recommendation.
- Save model identifier, generation timestamp, training cutoff, data cutoff, class mapping,
  blend metadata, and data-quality warnings with traceable fixture IDs.
- Use timezone-aware timestamps when available; document date-only source limitations.

**Proposed outputs:**
- `data/processed/upcoming_predictions.parquet` — latest forecast view.
- Immutable, run-versioned prediction snapshots under an agreed processed-data subdirectory.

**Acceptance criteria:**
- [ ] Upcoming fixtures survive inference with no target values.
- [ ] Valid finite, non-negative three-class probabilities sum to one within tolerance.
- [ ] Feature order matches the fitted model's saved schema exactly.
- [ ] Current and historical model versions are distinct and traceable.
- [ ] Forecast table is usable locally in the notebook; no UI dependency.
- [ ] Snapshot is saved before results become available and is not overwritten by later runs.

#### 5.3d — Evaluate Saved Forecasts After Matches Finish

**Requirements:**
- Join actual outcomes onto saved pre-match snapshots by stable fixture ID.
- Score the original saved forecast, not a forecast regenerated after the result.
- Define which snapshot is evaluated when multiple pre-match forecasts exist.
- Leave unplayed/postponed fixtures unscored and report their status.
- Preserve original probabilities, timestamps, and model metadata.

**Acceptance criteria:**
- [ ] Completed forecast snapshots can be scored without rerunning inference.
- [ ] Unplayed fixtures are excluded from scoring with explicit counts.
- [ ] Duplicate snapshot selection and fixture reconciliation rules are deterministic.

#### Task 5.3 — Shared Quality Gates

- [x] Prompt 2 policy path uses the versioned
      `disabled_no_verified_prematch_data` absence policy consistently for
      historical reconstruction and prospective preparation: zero applied
      counts, base-equal adjusted lambdas, and MC inputs from those safe
      lambdas. Legacy absence adjustment remains available when the policy is
      not selected.
- [x] Prompt 2 feature/rating history protections exclude same-date results
      when only date-level ordering is available.
- [ ] Unit tests use deterministic synthetic data and mocked/cached provider responses.
- [ ] Tests cover chronological cutoffs, same-date boundaries, upcoming null targets,
      exact feature order, save/load prediction parity, unknown teams, insufficient history,
      probability integrity, and outcome/snapshot alignment.
- [ ] Notebook executes with `pytest --nbval-lax`; live-network cells are skipped in CI.
- [ ] Focused tests, full suite, and `git diff --check` pass.
- [ ] `JUSTIFICATION.md` records fixed-model evaluation, as-of constraints, current-model
      refitting, missing-data policies, and snapshot design using actual implementation details.
- [ ] README documents how to refresh data, evaluate recent fixtures, and forecast upcoming games.
- [ ] Mark Task 5.3 complete only after both recent-match evaluation and upcoming inference work.

---

### P3 — Task 5.4: Dashboard (Deferred; formerly Task 5.3)
**Status:** Deferred — not a dependency for Task 5.3.
**Potential module:** `dashboard/app.py`  
**Potential tests:** `tests/dashboard/test_app.py`

**Goal:** Optionally present validated ratings, forecasts, and evaluation history in a UI.

**Requirements:**
- Decide whether a Streamlit dashboard is useful only after the notebook prediction workflow works.
- If approved, read existing processed artifacts; do not duplicate model or feature logic.
- Display current team ratings, upcoming predictions, Brier trend, and data/model freshness.

**Acceptance criteria (only if activated):**
- [ ] User confirms a dashboard is wanted.
- [ ] App runs locally and displays the agreed components.
- [ ] Basic rendering tests pass.
- [ ] JUSTIFICATION entry explains dashboard design.

---


## Summary by Priority

| Priority | Tasks |
|----------|-------|
| P0 | 1.1 (Ingestion via soccerdata), 1.2 (Cleaning), 1.3 (Poisson ratings) |
| P1 | 2.1 (Absence classification), 2.2 (λ adjustment), 3.1 (Scoreline matrix), 3.2 (Monte Carlo) |
| P2 | 4.1 (Feature table), 4.2 (XGBoost), 4.3 (Ensemble), 5.1 (Calibration evaluation), 5.2 (Recalibration experiment), 5.3 (Recent evaluation & upcoming prediction) |
| P3 | 5.4 (Optional dashboard — deferred) |

Start with P0 tasks in order — each depends on the previous one's output.