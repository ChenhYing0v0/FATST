"""Reuse audited rendering on a separate balanced-case output directory."""
import importlib.util
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "builder", BASE / "ettm1_segments_20260906/build_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.main()


if __name__ == "__main__":
    main()
