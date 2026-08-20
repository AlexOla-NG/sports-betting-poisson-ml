2026-08-15 — Added agent customization guidance files and CI check script to validate agent docs. Changes:

- Add `AGENTS.md` and `.github/copilot-instructions.md` for agent guidance
- Add `scripts/agent_customize_checks.py` and GitHub Actions workflow to run checks

Rationale: Ensure agent customization files follow repository conventions and are validated by CI before merging.

# Design Justification Log

This file records every significant design decision, the alternatives
considered, and the reasoning behind the final choice. Update this file
whenever a new feature is added or an existing design changes.

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
