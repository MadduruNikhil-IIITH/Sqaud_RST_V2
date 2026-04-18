# Phase 6: score_feature_salience

## Goal
Compute feature-based salience score, rank, and label from the Phase 5 feature table.

## Command
python -m src.pipeline.run_stage --stage score_feature_salience

## Inputs
- data/processed/feature_table.csv

## Outputs
- data/processed/feature_table_scored.csv
- data/interim/feature_score_report.json

## Method
1. Normalize numeric features with z-score.
2. Compute weighted linear score as feature_salience_score.
3. Rank by paragraph (para_id) to generate feature_salience_rank.
4. Create feature_salience_label from top-K per paragraph.

## Success criteria
- One scored row per input feature row.
- feature_salience_score present for all rows.
- feature_salience_rank present for all rows.
- feature_salience_label present for all rows.
- Report file contains basic classification metrics when gold labels are available.
