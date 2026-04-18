# Phase 2: segment_sentences

## Goal
Create sentence table with paragraph-coordinate offsets and strict alignment checks.

## Command
python -m src.pipeline.run_stage --stage segment_sentences

## Outputs
- data/processed/sentence_table.csv
- data/interim/segmentation_diagnostics.json

## Success criteria
- All sentences reconstructed correctly from offsets
- No failed alignments
