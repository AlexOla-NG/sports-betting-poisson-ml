# GitHub guidance for Copilot and agents

Use [AGENTS.md](../AGENTS.md) for repository-wide agent behavior and
[CONVENTIONS.md](../CONVENTIONS.md) for the detailed modeling policy. This file
keeps only checks that are especially useful when preparing a GitHub change.

Before opening a PR:
- Run `pytest -q` and `pytest --nbval-lax` from the repository root.
- Run `python3 scripts/agent_customize_checks.py` when changing agent guidance.
- New `src/` modules need docstrings, type hints, and a matching test under [tests/](../tests/); mock soccerdata and do not read committed data artifacts.
- Notebook edits need an inputs/outputs top cell, grouped imports, markdown section headers, and `#NBVAL_SKIP` or `#NBVAL_IGNORE_OUTPUT` for live or intentionally variable cells.
- Resolve notebook paths from the repository root, not from a fixed current-working-directory parent chain. Downstream notebooks should fail clearly when required Parquet inputs are missing.
- Keep persisted features at one row per fixture, use config-driven tunables, and preserve `.shift(1)` point-in-time behavior. Internal two-row GLM inputs are acceptable only when the persisted output remains one row per fixture.
- Record design decisions in [JUSTIFICATION.md](../JUSTIFICATION.md) and update [README.md](../README.md) only when scope or structure changes.

Do not commit data, test output, virtual environments, cache files, credentials,
or secrets. Do not create commits unless the user explicitly requests one.
