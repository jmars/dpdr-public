"""Alias driver: `.venv/bin/python -m experiments.exp2`."""
from .exp2_rescue import main

if __name__ == "__main__":
    raise SystemExit(main())
