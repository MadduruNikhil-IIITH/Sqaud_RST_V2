# Phase 5+ Roadmap

## Phase 5: feature_extraction

## Goal
Build a sentence-level feature table by merging segmentation, gold labels, and RST links, with optional PsychFormers surprisal features.

## Command
python -m src.pipeline.run_stage --stage feature_extraction

Optional PsychFormers run:
python -m src.pipeline.run_stage --stage feature_extraction --run-psychformers --psychformers-dir <path-to-psychformers> --psychformers-model gpt2 --psychformers-decoder masked --psychformers-following-context

## Outputs
- data/processed/feature_table.csv
- data/interim/psychformers/stims/sentence_level.stims
- data/interim/psychformers/stims/word_level.stims
- data/interim/psychformers/output/*.output (when PsychFormers is run)

## Generated features
- rst_relation
- rst_nuclearity
- rst_tree_depth
- span_importance_score
- sentence_position_ratio
- named_entity_count
- prev_next_cohesion_score
- paragraph_discourse_continuity_score
- cue_word_flags
- content_word_density
- sentence_length_tokens
- lexical_density
- surprisal_sentence_total (optional, PsychFormers)
- surprisal_sentence_per_token (optional, PsychFormers)
- surprisal_word_mean (optional, PsychFormers)
- surprisal_word_max (optional, PsychFormers)
- surprisal_word_std (optional, PsychFormers)

## Success criteria
- One feature row per sentence in data/processed/sentence_table.csv
- sent_id/para_id/sent_idx alignment remains stable
- RST-derived fields populated when parse status is success
- PsychFormers columns are populated when output files are available

## Planned next stages
1. feature_extraction
2. score_feature_salience
3. annotate_llm_salience
4. evaluate_salience

## Phase 6: score_feature_salience

## Goal
Compute feature-based salience score, paragraph-wise rank, and binary label from the Phase 5 feature table.

## Command
python -m src.pipeline.run_stage --stage score_feature_salience

## Outputs
- data/processed/feature_table_scored.csv
- data/interim/feature_score_report.json

## Scoring approach
- Global z-score normalization per numeric feature.
- Weighted linear blend to compute feature_salience_score.
- Rank sentences within each para_id by descending score.
- Assign feature_salience_label to top-K per paragraph where K defaults to gold positive count in that paragraph (or 1 if unavailable).

## Success criteria
- feature_salience_score populated for all rows.
- feature_salience_rank populated for all rows.
- feature_salience_label populated for all rows.
- Report JSON generated with basic precision/recall/F1 when gold_salient exists.

## Processed salience model
Use src/common/models.py ProcessedSalienceRecord as canonical merged row:
- sent_id
- para_id
- sent_idx
- gold_salient
- feature_score
- feature_rank
- llm_label
- llm_score
- llm_rank
