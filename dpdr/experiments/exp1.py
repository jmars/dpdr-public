"""Alias driver: `.venv/bin/python -m experiments.exp1`."""
from .exp1_failure_threshold import main

if __name__ == "__main__":
    raise SystemExit(main())
