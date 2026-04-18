# Phase 4: parse_rst_trees

## Goal
Generate paragraph-level RST artifacts and tree image manifest.

## Command
python -m src.pipeline.run_stage --stage parse_rst_trees

## Outputs
- data/artifacts/rst_artifacts.jsonl
- data/artifacts/rst_image_manifest.csv

## Success criteria
- One rst artifact row per paragraph
- One image manifest row per paragraph
