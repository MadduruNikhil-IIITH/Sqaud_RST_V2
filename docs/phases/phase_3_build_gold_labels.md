# Phase 3: build_gold_labels

## Goal
Map answer spans to sentence ids and generate gold salience labels.

## Command
python -m src.pipeline.run_stage --stage build_gold_labels

## Outputs
- data/processed/answer_sentence_map.jsonl
- data/processed/gold_sentence_salience.csv

## Success criteria
- Every sentence has gold label row
- Mapping status recorded for each answer
