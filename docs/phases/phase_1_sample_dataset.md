# Phase 1: sample_dataset

## Goal
Select deterministic 15-paragraph SQuAD v2.0 slice from HuggingFace datasets with traceability metadata.

## Command
python -m src.pipeline.run_stage --stage sample_dataset

## Output
- data/interim/sample_manifest.json

## Success criteria
- Exactly 15 paragraph records sampled
- Stable seed-driven sampling (deterministic)
- Data fetched automatically from HuggingFace datasets
