"""Validate the minimal R_2026_FATST repository skeleton."""

from pathlib import Path


REQUIRED_PATHS = [
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "analysis",
    "artifacts",
    "baselines",
    "configs",
    "data",
    "docs",
    "docs/code-explanation",
    "docs/experiments",
    "docs/remote",
    "Papers",
    "scripts",
    "scripts/remote",
    "src/fatst",
    "tests",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_PATHS if not (repo_root / path).exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing required project paths:\n{formatted}")
    print("Project structure check passed.")


if __name__ == "__main__":
    main()
