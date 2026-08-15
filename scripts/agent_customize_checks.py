#!/usr/bin/env python3
"""Simple agent customization checks.

Usage: `python3 scripts/agent_customize_checks.py`

Checks performed:
- `AGENTS.md` exists at repo root
- `.github/copilot-instructions.md` exists
- `README.md` contains a link to `AGENTS.md`
- `src/config/JUSTIFICATION.md` contains a recent entry for today's date

This script is intentionally lightweight and dependency-free so reviewers
or CI jobs can run it without extra setup.
"""
import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]

def main():
    failures = []
    if not (ROOT / "AGENTS.md").exists():
        failures.append("AGENTS.md not found at repo root")
    if not (ROOT / ".github" / "copilot-instructions.md").exists():
        failures.append(".github/copilot-instructions.md not found")
    readme = ROOT / "README.md"
    if not readme.exists() or "AGENTS.md" not in readme.read_text(encoding="utf8"):
        failures.append("README.md does not reference AGENTS.md")
    just = ROOT / "src" / "config" / "JUSTIFICATION.md"
    today_tag = f"## {date.today().isoformat()}"
    if not just.exists() or today_tag not in just.read_text(encoding="utf8"):
        failures.append(f"JUSTIFICATION.md missing an entry starting with '{today_tag}'")

    if failures:
        print("Agent customization checks failed:")
        for f in failures:
            print(" - ", f)
        sys.exit(2)
    print("All agent customization checks passed.")

if __name__ == '__main__':
    main()
