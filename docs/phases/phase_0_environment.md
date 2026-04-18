# Phase 0: Environment Gate

## Goal
Create a reproducible GPU-ready environment and verify core package gates.

## Inputs
- environment.py312.yml
- environment.py311.yml
- environment.py310.yml

## Commands
1. conda env create -f environment.py312.yml
2. python scripts/validate_environment_gate.py
3. Fallback to py311 and py310 if needed.

## Success criteria
- Python runtime available
- torch import works
- cuda visibility reported
- isanlp-rst import works
