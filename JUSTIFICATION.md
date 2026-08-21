# Design Justification Log

This file records every significant design decision, the alternatives
considered, and the reasoning behind the final choice. Update this file
whenever a new feature is added or an existing design changes.

## 2026-08-21 — Task 1.3 Poisson Attack & Defense Ratings Implementation

**Decision:** Implement `src/ratings/poisson.py` and `notebooks/03_ratings/01_poisson_ratings.ipynb` using a log-linear Poisson GLM (`statsmodels.api.GLM` with Log link) fitted over a per-team rolling match window (`rating_window = 8` from `config.yaml`) with strict `.shift(1)` point-in-time correctness, expanding window cold start (defaulting to 1.0 multiplier), and normalized attack/defense ratings centered around 1.0.

**Alternatives considered:**
1. Dixon-Coles adjusted Poisson model: Deferred to model refinement stage since basic Poisson GLM provides a clean, robust baseline for attack/defense parameter estimation without requiring numerical optimization for low-score tau adjustments.
2. Fixed date window (e.g. 60 days): Rejected in favor of a fixed match-count window per team (`rating_window`) to handle match postponements and irregular scheduling evenly across teams.
3. Static seasonal fit: Rejected because static fits cause severe forward data leakage; point-in-time rolling GLMs ensure match $k$'s outcome never leaks into match $k$'s pre-match rating.

**Rationale:** Poisson GLM with home-field advantage provides separate, interpretable attack and defense multipliers per team. Normalizing parameters to mean 1.0 simplifies downstream expected goal ($\lambda$) calculations ($\lambda_{\text{home}} = \text{league\_avg} \times \text{att}_{\text{home}} \times \text{def}_{\text{away}} \times \sqrt{\text{home\_adv}}$).


---

## 2026-08-21 — Task 1.2 Data Cleaning & Base Features Implementation

**Decision:** Implement data cleaning module `src/processing/clean_features.py` and notebook `notebooks/02_processing/01_clean_features.ipynb` using explicit team-name mapping dictionaries, 1-row-per-fixture layout, point-in-time non-leaking differential features, and explicit missing xG indicator flags (`is_xg_missing`).

**Alternatives considered:**
1. Dynamic fuzzy matching for team names: Rejected because hardcoded dictionary mapping for EPL team names between FBref and ClubElo is 100% deterministic, instant, and eliminates runtime ambiguity.
2. Imputing missing xG with global or team rolling mean during cleaning: Rejected because imputing raw data during base feature cleaning creates potential data leakage if unplayed/postponed fixtures use future statistics. Retaining `NaN` and flagging `is_xg_missing` preserves data integrity for point-in-time downstream rolling models.
3. 2-row-per-fixture layout (one per team): Rejected per repository hard constraint (Rule 1 in CONVENTIONS.md: 1 row per fixture, home/away diff features).

**Rationale:** Standardizing team names guarantees clean joins with ClubElo in later ML stages. Producing home minus away differential features directly aligns with the single-row fixture layout and walk-forward validation requirements.


---

## 2026-08-21 — Auto-Sanitize Numeric Types in Match Stats Ingestion

**Decision:** Add `_sanitize_match_stats` and `fetch_all_match_stats` to `SoccerDataClient` to auto-detect and coerce numeric-looking object columns (e.g. `GF`, `GA`, `xG`, `Sh`) to nullable integer (`Int64`) or float (`Float64`) types.

**Alternatives considered:**
1. Hardcode column list for type coercion: Rejected because FBref returns varied table columns across endpoints and endpoints may change column schema; auto-detection is resilient without maintaining column lists.
2. In-line notebook coercion: Rejected because type cleaning at ingestion prevents downstream Parquet write errors across all execution contexts.
3. Coerce to float64 for all numeric columns: Rejected because integer counts like goals scored (`GF`/`GA`) lose integer semantics when converted to float.

**Rationale:** FBref table columns frequently contain mixed types (integers mixed with empty string placeholders or non-numeric strings for postponed/unplayed matches). PyArrow fails with `ArrowTypeError` when attempting to serialize object columns with mixed types to Parquet. Auto-detecting numeric object columns (where >=80% of non-blank entries are numeric) and casting to `Int64` or `Float64` ensures clean Parquet serialization while preserving nullable `NaN` values.


---

## 2026-08-20 — Defer Player Availability Retrieval

**Decision:** Exclude per-fixture player-minute retrieval from the base ingestion notebook and defer it to the targeted absence-classification stage.

**Alternatives considered:**
1. Retrieve player match pages for every fixture during ingestion: Rejected because it requires hundreds of sequential dynamic-page requests and can take hours.
2. Infer injury, suspension, and rotation reasons directly from minutes: Rejected because minutes show participation, not the reason for non-participation.
3. Remove player availability support entirely: Rejected because Task 2.1 may need targeted player logs for regular starters and high-minute players.

**Rationale:** Fixtures and team match statistics are sufficient for cleaning and Poisson ratings. The later absence stage can first identify regular players from season-level data, then retrieve only relevant match logs and classify results conservatively as availability states. The existing client method remains available for that targeted workflow without slowing base ingestion.

---

## 2026-08-20 — Migrate Ingestion To soccerdata

**Decision:** Replace the custom website scraper with a thin `soccerdata` adapter using FBref for fixtures and match/player statistics, and ClubElo as an additive rating input for later ML features.

**Alternatives considered:**
1. Continue maintaining the previous custom scraper: Rejected because its undocumented routes and anti-bot behavior caused repeated 404 and maintenance issues.
2. Use `statsbombpy`: Rejected because its free historical coverage does not provide the required season-over-season EPL dataset.
3. Use `understatapi`: Deferred because shot-level xG data is not required for the current Poisson, Monte Carlo, and XGBoost stages.
4. Replace Poisson ratings with ClubElo: Rejected because Elo and Poisson ratings represent different signals; Poisson remains the domain-specific primary model and Elo becomes an ML feature.

**Rationale:** `soccerdata` provides maintained FBref and ClubElo readers, caching, standardized Pandas outputs, and rate-limit handling. The wrapper preserves the existing fixture/statistics method names and Parquet handoff while removing website-specific HTTP and HTML parsing from this repository. The FBref league name is configurable because FBref uses league names rather than provider-specific numeric IDs.

---

## 2026-08-20 — Initial Ingestion Pipeline Implementation

**Decision:** Implement the initial ingestion pipeline with separate Parquet tables (fixtures, match_stats, player_minutes), resilient retrieval, single-season initial scope, and notebook-driven orchestration.

**Alternatives considered:**
1. Single denormalized table (all fixtures + stats + minutes flattened): Rejected because it would require duplicating fixture data for each match stat/minute record, increasing storage and making it harder to update any one table independently.
2. Database (SQLite/PostgreSQL) instead of Parquet: Rejected because Parquet is simpler for a single-developer ML pipeline, works well with pandas, and avoids deployment complexity.
3. Linear retry (fixed delay) instead of exponential backoff: Rejected because exponential backoff is the industry standard for API resilience; avoids overwhelming the server if the API is struggling.
4. Multi-season ingestion in one notebook run: Rejected for now; single-season keeps initial implementation simple and allows debugging of one season at a time.
5. Python script (not notebook) for ingestion: Rejected because notebooks allow interactive exploration, easier debugging of data issues, and align with the project's narrative-driven pipeline approach per CONVENTIONS.md.

**Rationale:**
- **Separate tables:** One row per fixture (or match stat row, or player minute row) avoids data duplication and allows independent updates. Fetching is also faster and clearer.
- **Parquet format:** Efficient compression, native pandas support, and self-describing schema. Ideal for iterative ML development.
- **Resilient retrieval:** The data library's caching and request handling reduce transient network failures without custom website-specific logic.
- **Notebook-first:** Aligns with project structure (7-stage pipeline of notebooks); allows downstream stages to load `data/raw/` Parquet as input.
- **Single season:** 2024/2025 season only (for now). Simplifies initial development; multi-season loop can be added later if needed.
- **Error logging + continue:** If a single fixture fails, log and move on rather than halting the entire ingestion. Critical errors (no fixtures at all, auth failure) still fail fast.

---


## 2026-08-15 — League & Season Scope

**Decision:** Start with EPL only, 2 historical seasons.

**Alternatives considered:**
- Multiple leagues from day one — rejected: increases data-cleaning complexity
  (different stat availability per league) before the core pipeline is validated.
- 1 season only — rejected: insufficient backtesting depth for walk-forward
  validation to produce a stable Brier score trend.

**Rationale:** Validate the full pipeline (ingestion → ratings → ML → evaluation)
on a single, data-rich league before scaling. EPL has strong FBref data coverage
(xG, shots, lineups). Architecture will remain league-agnostic (league name as a
parameter) to support future expansion.

---

## 2026-08-15 — Row Structure: One Row Per Fixture

**Decision:** One row per fixture with home/away diff features, not two rows
per fixture (team-perspective).

**Alternatives considered:**
- Two rows per fixture (one per team) — rejected: doubles dataset size without
  adding information, and loses the direct home-vs-away comparison structure
  that diff features encode naturally.

**Rationale:** Diff-based single-row structure is simpler to reason about,
reduces feature count, and matches the structure used in comparable reference
implementations for football outcome prediction.

---

## 2026-08-15 — Rolling Window Lengths: Tunable Hyperparameters

**Decision:** Rolling window lengths for form and xG features are not
hardcoded; they live in `src/config/config.yaml` and are treated as hyperparameters.

**Alternatives considered:**
- Fixed windows (5 for form, 8 for xG) based on literature priors — rejected
  as the final choice, though retained as starting priors.

**Rationale:** Literature converges on a 5-10 match range for both metrics,
with the "right" window varying by dataset and league. Rather than lock in a
single value, expose window length as a config parameter so it can be tuned
empirically via walk-forward validation once real backtest data exists.

**Starting priors (subject to tuning):**
- Points/goals form: 5-6 matches (outcome-based, noisier, needs faster reaction)
- xG for/against: 6-10 matches (lower-variance, shot-quality metric, benefits
  from more data before stabilizing)
- Attack/defense ratings: 8 matches (balance between stability and responsiveness)

---

## 2026-08-15 — Development Environment & Tooling

**Decision:** VS Code as primary editor, with Aider
(`ollama/qwen2.5-coder:7b`) for AI-assisted development. Manual notebook/script
execution for now; no orchestrator (Airflow/Prefect) until recurring job count
justifies it.

**Rationale:** Weekly gameweek cadence does not require scheduling
infrastructure at this stage. Local LLM tooling (Aider + Ollama) keeps
development cost-free and code private, at the tradeoff of smaller model
capacity — mitigated by scoping Aider's context to relevant files per task
rather than the whole repo.

---

## 2026-08-15 — Notebook Testing Standard

**Decision:** All notebooks must have tests via `pytest --nbval-lax`.

**Alternatives considered:**
- Strict `--nbval` (exact output matching) — rejected as the default, since
  notebook outputs (ratings, simulation results) legitimately change as new
  data is ingested.

**Rationale:** Lax mode validates that notebooks execute without error, which
is the meaningful test for exploratory/data pipeline notebooks, without being
brittle to expected output drift. Cells with live API calls or non-deterministic
output should be marked `#NBVAL_SKIP` or `#NBVAL_IGNORE_OUTPUT`.

---

## 2026-08-15 — File Format: Notebooks + src/ Split

**Decision:** Pipeline code is written as Jupyter notebooks (.ipynb) for
orchestration and exploration, with shared logic factored out into thin,
tested modules under `src/` that notebooks import from.

**Alternatives considered:**
- Pure notebooks (all logic inline in .ipynb cells) — rejected: leads to
  duplicated logic across notebooks, harder to unit-test and maintain.
- Pure scripts (.py only) — rejected: loses the narrative, exploratory benefit
  of notebooks for data analysis and visualization.

**Rationale:** Hybrid approach: notebooks focus on orchestration, visualization,
and narrative flow; `src/` modules hold reusable logic (rating calculations,
injury adjustment formulas, Monte Carlo sampling) that can be unit-tested
independently. This keeps notebooks readable while avoiding code duplication.

---

## 2026-08-15 — Persistent AI Context via CONVENTIONS.md

**Decision:** Use Aider's `CONVENTIONS.md` + `.aider.conf.yml` mechanism to
load project conventions automatically into every session, rather than pasting
a master prompt each time.

**Alternatives considered:**
- Paste full context block manually every session — rejected: error-prone,
  easy to forget, wastes tokens.
- Rely on model memory across sessions — rejected: Aider resets every session;
  no persistent memory except via explicitly loaded files.

**Rationale:** Aider natively supports loading a read-only conventions file
every session via `--read` or `.aider.conf.yml`. This encodes all hard
constraints (diff-based rows, no hardcoded windows, no leakage, walk-forward
validation, mandatory tests, documentation obligation) in a single source of
truth that the model can't accidentally edit or ignore.

---

## 2026-08-15 — Centralized Configuration (src/config/config.yaml)

**Decision:** All hyperparameters (rolling windows, injury weights, XGBoost
params, ensemble weights, Monte Carlo trials) live in a single YAML file,
loaded via `src/config/loader.py`.

**Alternatives considered:**
- Hardcode windows/weights inline in notebooks/modules — rejected: makes
  hyperparameter tuning painful (hunt-and-replace across files).
- Python dataclass only — rejected: YAML is easier to sweep programmatically
  in backtests and easier to edit without touching code.

**Rationale:** Single source of truth for all tunable knobs. Enables systematic
hyperparameter sweeps (e.g., testing form_window in [5,6,7,8]) without modifying
code. The loader module provides convenience accessors so notebooks don't
navigate nested dicts everywhere.

---

## 2026-08-15 — Add AGENTS.md and GitHub copilot instructions

**Decision:** Add a top-level `AGENTS.md` and a `.github/copilot-instructions.md`
to provide concise, actionable guidance for AI agents and Copilot within PR
workflows. Also add `scripts/agent_customize_checks.py` to verify these files
are present and referenced by `README.md` and `src/config/JUSTIFICATION.md`.

**Alternatives considered:**
- Only add `AGENTS.md` (rejected: GitHub PR reviewers and CI benefit from a
  small `.github/copilot-instructions.md` optimized for PR checks).
- Embed guidance only in `README.md` (rejected: separation of concerns — a
  dedicated AGENTS.md keeps agent guidance discoverable and small).

**Rationale:** Explicit, minimal agent guidance reduces accidental
violations of project constraints (notebooks-first, no hardcoded windows,
walk-forward validation). The check script provides a lightweight CI hook
maintainers can run to ensure agent docs are present and referenced.

---

## 2026-08-15 — Ingestion: Parquet + Resilient Retrieval

**Decision:** Save raw ingestion outputs as Parquet files and rely on the
data-library cache and request handling for resilient retrieval.

**Alternatives considered:**
- JSON/NDJSON files — rejected: slower I/O for tabular analytics and larger on-disk
  footprint for repeated experimental runs.
- Storing raw responses in a DB (SQLite/Postgres) — rejected for early-stage work
  to avoid ops complexity; revisit when dataset size or concurrency demands it.

**Rationale:** Parquet is columnar, compact, and fast to read with `pandas` and
other analytics tools; it supports schema evolution and avoids repeated parsing
during iterative development. Delegating caching and request handling to the
maintained data library keeps the adapter lightweight and testable.

---

## 2026-08-15 — Added agent customization guidance files and CI check script to validate agent docs. Changes:

- Add `AGENTS.md` and `.github/copilot-instructions.md` for agent guidance
- Add `scripts/agent_customize_checks.py` and GitHub Actions workflow to run checks

**Rationale:** Ensure agent customization files follow repository conventions and are validated by CI before merging.
