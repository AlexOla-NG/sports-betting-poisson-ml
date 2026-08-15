# AGENTS.md — Guidance for AI coding agents

Purpose: provide concise, actionable instructions to AI coding agents working in this repository so they can be immediately productive and follow project conventions.

Quick links
- CONVENTIONS: [CONVENTIONS.md](CONVENTIONS.md)
- Config: [config/config.yaml](config/config.yaml)
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

Test and validation commands (use these in CI and locally)
- Run unit tests: `pytest -q`
- Validate notebooks: `pytest --nbval-lax`
- Agent docs check (local): `python3 scripts/agent_customize_checks.py`

Agent behavior: do this
- Prefer editing or adding small `src/` modules rather than embedding large functions inline in notebooks.
- When editing/adding files, include docstrings with inputs/outputs and pipeline stage, and add a pytest file under `tests/` for that module.
- For any change that affects repo structure or design decisions, add an entry to `src/config/JUSTIFICATION.md` (see [CONVENTIONS.md](CONVENTIONS.md)).
- When encountering ambiguous design choices, STOP and ask a clarifying question rather than guessing.
- Always update `README.md` and `CONVENTIONS.md` when scope or structure changes.

Agent behavior: do NOT do this
- Do not hardcode hyperparameters (window lengths, etc.) — read them from `config/config.yaml`.
- Do not commit large data files under `data/` — data is gitignored.
- Do not run or commit live API credentials. Mark live-API notebook cells with `#NBVAL_SKIP` or `#NBVAL_IGNORE_OUTPUT`.
- Do not modify `CONVENTIONS.md` without discussing with the maintainer; it's used as persistent AI context.

Where to look first
- Start with [CONVENTIONS.md](CONVENTIONS.md) and `config/config.yaml` for hyperparameter and pipeline rules.
- For tests and examples, inspect `tests/` and `notebooks/`.

If you plan to add new agent customizations (skills, prompts, hooks)
- Keep them minimal and link to existing docs instead of copying content.
- Name new files clearly (e.g., `AGENTS.md`, `.github/copilot-instructions.md`) and preserve existing content when updating.
- Add a short JUSTIFICATION.md entry describing the customization and rationale.

If you have questions or want to propose additional agent hooks, ask the maintainer and include a short justification entry for `src/config/JUSTIFICATION.md`.
