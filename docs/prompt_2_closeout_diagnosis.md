# Prompt 2 Closeout Diagnosis

**Date:** 2026-11-07
**Scope:** Read-only preservation/provenance diagnosis of the saved blend-contract
failure and current-season Understat xG retrieval. No notebooks were executed
and no production artifacts were regenerated.

## Blend-contract failure

The saved calibration and recalibration notebook log reports:

```text
ValueError: selected weights must be constant 0.80 XGBoost and 0.20 MC
```

The failing condition is in `validate_selected_ensemble_frame()` and is
reused by `validate_recalibration_input()`:

```python
not np.isclose(predictions["selected_xgb_weight"], 0.8, atol=1e-12).all()
or not np.isclose(predictions["selected_mc_weight"], 0.2, atol=1e-12).all()
```

The failed notebook cells read
`data/processed/ensemble_probabilities.parquet`. The configured contract is
`selected_candidate = "blend_context"`, XGBoost weight `0.8`, and MC weight
`0.2`, from `src/config/config.yaml`.

Current read-only artifact inspection:

| Artifact | Rows | Schema summary | Candidate/weights | Nulls and IDs | Mtime |
|---|---:|---|---|---|---|
| `ensemble_sweep_results.parquet` | 45 | candidate, weights, OOF count, metrics, selection flag | selected row: `blend_context`; XGB `0.85`; MC `0.15` | no nulls in inspected columns; 45 rows | 2026-11-07 20:45:34 UTC |
| `task_4_3_evaluation_report.parquet` | 10 | audit summary plus candidate rows and validation fields | audit summary: `blend_context`; XGB `0.8`; MC `0.2` | audit summary reports 560 selected/unique fixtures and no invalid probability counts | 2026-10-10 21:31:27 UTC |
| `ensemble_probabilities.parquet` | 560 | 9 columns: fixture/date/target, 3 ensemble probabilities, candidate, 2 weights | `blend_context`; XGB `{0.85}`; MC `{0.15000000000000002}` | no nulls in inspected contract columns; 560 unique fixture IDs | 2026-11-07 20:45:34 UTC |
| `recalibrated_probabilities.parquet` | 360 | 16 columns including raw/calibrated probabilities and fold metadata | no candidate/weight columns | no nulls in inspected IDs/target; 360 unique fixture IDs | 2026-11-04 06:23:43 UTC |
| `xgboost_probabilities.parquet` | 760 | 11 columns including XGBoost probabilities and targets | not applicable | no nulls in inspected IDs/target; 760 unique fixture IDs | 2026-11-07 20:45:28 UTC |
| `mc_probabilities.parquet` | 760 | 15 columns including lambdas, absence counts, and MC probabilities | not applicable | no nulls in fixture IDs; 760 unique fixture IDs | 2026-11-07 20:45:03 UTC |

All seven requested existing artifacts were copied byte-for-byte before
inspection. The preservation manifest and copies are in
`data/processed/diagnostics/prompt_2_closeout_20261107T210128Z/`; it records
source paths, sizes, modification times, and SHA-256 hashes. No requested
artifact was missing.

The pure read-only calls
`validate_selected_ensemble_frame(pd.read_parquet(...))` and
`validate_recalibration_input(pd.read_parquet(...))` both reproduce the same
weight-contract exception. This establishes the current artifact is
incompatible with the current validator/config contract. Modification times
alone do not establish which process produced the `0.85/0.15` artifact or
whether it overwrote the state used by the earlier notebook failure.

Independently recomputed metrics for the current 560-row ensemble are accuracy
`0.4928571428571429`, log loss `1.0261158920935058`, and Brier score
`0.6142683959116398`; these match the current 85/15 sweep row. The persisted
component tables have matching unique fixture IDs, but
`xgboost_probabilities.parquet` is explicitly the full-XGBoost table while the
notebook builds a separate in-memory context-XGBoost table for
`blend_context`. Reconstructing from the persisted full-XGBoost and MC tables
does not reproduce the current ensemble (maximum absolute error is
`0.11700688860808922` at 85/15). The selected context-only component is not
persisted, so the original 80/20 probability rows are unavailable/unverified;
no supposedly identical reconstruction was substituted.

Known saved notebook output contains both 80/20 and 85/15 candidate rows and
prints 80/20 as its selected value, while the current sweep and ensemble
artifacts select/store 85/15. This establishes disagreement across saved
selection artifacts, not the identity of the writer. No repository log
identified a writer.

## Understat xG retrieval

The installed environment is `.venv` with `soccerdata==1.9.1`. The failing
method is `soccerdata.understat.Understat.read_schedule()`, which calls
`_read_league_season()`. For the 2026 season, that method constructs the API
URL:

```text
https://understat.com/getLeagueData/EPL/2026
```

`_read_league_season()` calls `_request_api()`, then immediately executes
`json.load(reader)`. The existing cache entry is:

```text
~/soccerdata/data/Understat/league_1_season_2026.json
```

Read-only cache evidence:

- exists;
- 26,719 bytes;
- ASCII HTML document, not JSON;
- begins with `<!DOCTYPE html><html>`;
- contains the title `Understat.com`;
- ends with `We will come back soon`;
- JSON parsing fails with `JSONDecodeError: Expecting value: line 1 column 1`;
- cache mtime: 2026-11-07 20:13:25 UTC.

The installed `_request_api()` uses an existing cache whenever the cache file
exists and cache is enabled; otherwise it performs a GET, calls
`raise_for_status()`, and stores the response bytes at that path. The probe
manifest records the exception and traceback but does **not** record HTTP
status, response headers/content type, final URL, response size, or whether
the bytes came from cache versus a fresh request.

One authorized diagnostic GET was then made without reading or writing that
cache, with the installed `tls_requests` client, the Understat
`X-Requested-With: XMLHttpRequest` header, a 15-second timeout, automatic
redirect following disabled, and no retry loop. The response evidence is
saved in
`data/processed/diagnostics/prompt_2_closeout_20261107T210128Z/understat_diagnostic_get.json`:

- URL: `https://understat.com/getLeagueData/EPL/2026`;
- UTC retrieval: `2026-11-07T21:01:48.553517+00:00`;
- status `200`, content type `text/html`, size `26719` bytes;
- no `Location` header; final URL equals the requested URL;
- body begins with `<!DOCTYPE html>` and the Understat error-page markup;
- in-memory JSON parsing failed with `JSONDecodeError` at line 1, column 1.

The live response therefore confirms that this endpoint currently returns an
HTML service/error page rather than the JSON data expected by the installed
reader. It does not establish why Understat is serving that page, and no
current-season match/xG coverage is inspectable from it.

## Smallest approved next step

No repair was performed. The smallest next step requiring approval is a
decision about whether to pursue Understat access/reader repair; this session
does not justify a parser repair, cache replacement, provider change, or xG
substitution. The original 80/20 predictions remain unavailable/unverified,
and the current 85/15 artifact remains incompatible with the 80/20 validator
contract.

## Offline FBref xG check

The existing current-season probe
`data/processed/probes/probe_20261107_201301/fbref_schedule_2026_2027.parquet`
contains 380 rows with these columns and dtypes:

```text
league str, season str, game str, week Int64, day string, date datetime64[us],
time string, home_team string, score string, away_team string,
attendance Int64, venue string, referee string, match_report string,
notes Int64, fixture_id string, home_goals Float64, away_goals Float64
```

It has no `home_xg`, `away_xg`, `xG`, or equivalent expected-goals column.
The 91 completed fixtures span `2026-08-21` through `2026-11-06` and have
zero non-null xG values; the 289 upcoming fixtures also have zero. Non-null
fixture IDs are unique for the 91 completed rows, while all 289 upcoming rows
have null IDs (the apparent duplicate count is therefore repeated nulls, not
duplicate completed fixtures). Both team columns are complete with 20 unique
home and 20 unique away teams. Goals, week, and attendance values have no
negative or non-finite numeric values; expected nulls occur for upcoming goals
and attendance.

The installed `soccerdata==1.9.1` FBref reader maps source schedule columns
named `xG` and `xG.1` to `home_xg` and `away_xg` in `FBref.read_schedule()`.
The project wrapper's `_normalize_schedule()` does not drop those columns.
However, the existing cached FBref schedule responses for seasons 2024/25,
2025/26, and 2026/27 contain no xG field markers, and the current probe
contains none. This is not evidence that the wrapper discarded xG.

Historical local match-stat artifacts do contain an `xG` column with 760/760
non-null team rows for both 2024/25 and 2025/26. Their provenance is not
FBref-only: `SoccerDataClient.fetch_all_match_stats()` explicitly creates an
Understat reader, reads its schedule, and writes the joined values into the
`xG` column. The cleaning stage then merges those team rows into fixture-level
`home_xg` and `away_xg`. Thus historical xG is locally available, but the
current FBref schedule does not provide usable xG and Understat currently
returns HTML rather than the expected JSON. A consistent source covering both
historical training and newer evaluation is not established.

**Conclusion: C — no usable existing FBref xG.** The missing data is
current-season fixture-level xG for completed and upcoming rows, plus a
verified source-consistency decision for historical training versus newer
evaluation. No sources were merged, no zeros were substituted, and no
processed artifacts or configuration were changed. The preserved legacy
selection evidence remains unchanged: current sweep/ensemble artifacts are
85/15, older audit evidence is 80/20, and original context-only 80/20
predictions remain unavailable/unverified.
