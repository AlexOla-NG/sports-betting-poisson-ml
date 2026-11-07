# FBref Fallback v1 Evaluation

**Pipeline:** `fbref_fallback_v1`
**Scope:** Task 5.3b recent completed-match evaluation only
**Environment:** `.venv/bin/python`
**Source data:** local artifacts only; no network request was made

## Frozen protocol

- Training data: 760 completed historical fixtures from
  `data/processed/clean_fixtures.parquet`.
- Verified training cutoff: `2026-05-24`.
- Evaluation source:
  `data/processed/probes/probe_20261107_201301/fbref_schedule_2026_2027.parquet`.
- Eligible newer completed fixtures: 91.
- Evaluation dates: `2026-08-21` through `2026-11-06`.
- Probe exclusions: 289 upcoming rows with missing scores; zero pre-cutoff
  rows; zero completed rows excluded for invalid scores.
- Class order: Home `0`, Draw `1`, Away `2`.
- XGBoost parameters were read from the existing configuration and frozen.
- Identity post-processing was used; no recalibration or weight sweep was run.
- Absence policy: `disabled_no_verified_prematch_data`; applied absence counts
  are zero and adjusted lambdas equal base lambdas.

## Frozen feature schema

The 15 ordered XGBoost inputs are:

```text
fixture_week
home_points_form, away_points_form
home_goals_for_form, away_goals_for_form
home_goals_against_form, away_goals_against_form
points_form_diff, goals_for_form_diff, goals_against_form_diff
rest_days_home, rest_days_away, rest_days_diff
h2h_home_win_rate, h2h_match_count
```

Features use strictly earlier completed history. Same-date fixture blocks are
scored before any result from that date updates history. Cold-start values are
configured explicitly. No xG, xG-missing flags, appearances, absences, Elo,
Poisson/MC columns, or current-match statistics enter the XGBoost vector.

## Candidate results

| Candidate | Fixtures | Accuracy | Summed multiclass Brier | Log loss |
|---|---:|---:|---:|---:|
| Fallback XGBoost only | 91 | 0.395604 | 0.653988 | 1.081158 |
| Safe base-lambda MC only | 91 | 0.384615 | 0.700993 | 1.155023 |
| Fixed 80% fallback XGBoost / 20% MC | 91 | 0.384615 | 0.656428 | 1.083815 |

These are descriptive results for the FBref fallback population and reduced
feature specification. They are not comparable to the legacy xG-based
artifacts as if they used identical fixtures or features, and they do not
establish betting profitability or a production winner.

## Versioned outputs

The generated artifacts are kept separately under
`data/processed/fbref_fallback_v1/`:

- `fbref_fallback_xgboost.json`
- `fixture_predictions.parquet`
- `candidate_metrics.parquet`
- `metadata.json`

`metadata.json` records the ordered schema, frozen parameters, class
distribution (`Home=37`, `Draw=25`, `Away=29`), MC seed/trials, source hashes,
and exclusion counts. Legacy models, ensemble outputs, calibration outputs,
and historical raw/processed artifacts were not overwritten.

This does not complete overall Task 5.3. Upcoming inference and snapshot
reconciliation remain separate work.
