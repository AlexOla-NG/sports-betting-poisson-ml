#!/usr/bin/env python3
"""Simple CI check for agent customization files.

Checks that AGENTS.md and .github/copilot-instructions.md exist,
that README references JUSTIFICATION.md, and JUSTIFICATION.md contains a date entry.
"""
import os
import re
import sys


def exists(path):
    return os.path.exists(path)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    checks = []

    agents_md = os.path.join(repo_root, "AGENTS.md")
    copilot = os.path.join(repo_root, ".github", "copilot-instructions.md")
    justification = os.path.join(repo_root, "JUSTIFICATION.md")
    readme = os.path.join(repo_root, "README.md")

    checks.append(("AGENTS.md exists", exists(agents_md)))
    checks.append((".github/copilot-instructions.md exists", exists(copilot)))

    if exists(readme):
        content = read(readme)
        checks.append(("README mentions JUSTIFICATION.md", "JUSTIFICATION.md" in content))
    else:
        checks.append(("README exists", False))

    if exists(justification):
        jcontent = read(justification)
        date_like = re.search(r"\d{4}-\d{2}-\d{2}", jcontent)
        checks.append(("JUSTIFICATION.md contains YYYY-MM-DD date", bool(date_like)))
    else:
        checks.append(("JUSTIFICATION.md exists", False))

    failed = False
    for desc, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f"{status}: {desc}")
        if not ok:
            failed = True

    if failed:
        print("\nOne or more checks failed. Fix the above and re-run.")
        sys.exit(2)

    print("All checks passed.")


if __name__ == "__main__":
    main()
