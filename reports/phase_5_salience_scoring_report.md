# Phase 5 Salience Scoring Report

## Scope
This report explains:
1. How salience score is currently calculated from features.
2. Whether surprisal features are included.
3. Whether salient labels are required.

## Current state in code and outputs
1. Feature extraction is implemented and produces `data/processed/feature_table.csv`.
2. The table includes engineered feature columns plus PsychFormers surprisal columns.
3. `feature_salience_score`, `feature_salience_label`, and `feature_salience_rank` are present as schema fields but are not computed yet.
4. This means score computation is currently in "planned next stage" status (`score_feature_salience`).

## Are we including surprisal features?
Yes.

Current integrated surprisal columns:
1. `surprisal_sentence_total`
2. `surprisal_sentence_per_token`
3. `surprisal_word_mean`
4. `surprisal_word_max`
5. `surprisal_word_std`

These can and should be included as predictors in feature-based salience scoring.

## Recommended feature salience score calculation
A practical first-pass scoring rule:

1. Build numeric predictor vector per sentence with available columns:
- rst_tree_depth
- span_importance_score
- sentence_position_ratio
- named_entity_count
- prev_next_cohesion_score
- paragraph_discourse_continuity_score
- content_word_density
- sentence_length_tokens
- lexical_density
- surprisal_sentence_per_token
- surprisal_word_mean
- surprisal_word_max

2. Normalize each numeric feature across the dataset (z-score).

3. Compute a weighted linear score:
- feature_salience_score = sum(w_i * z_i)

4. Rank sentences within each paragraph by descending score:
- feature_salience_rank = rank(score, descending, per para_id)

5. Optional binary label from score:
- feature_salience_label = 1 if rank <= K (for example K=1 or K=2)
- else 0

## Do we need salient labels?
Short answer:
1. For unsupervised ranking: not strictly required.
2. For supervised calibration and evaluation: yes, you need labels.

Details:
1. Gold labels (`gold_salient`) are required to learn feature weights in a principled way (logistic regression, learning-to-rank, threshold calibration).
2. Gold labels are required to evaluate quality (precision, recall, F1, AP, nDCG, etc.).
3. If labels are missing, you can still produce heuristic scores and rankings, but you cannot reliably validate model quality.

## Recommended stage split
1. Phase 5 (`feature_extraction`): generate feature table with surprisal and structural features.
2. Phase 6 (`score_feature_salience`): compute score/rank/label columns.
3. Phase 7 (`annotate_llm_salience`): add LLM score/rank/label columns.
4. Phase 8 (`evaluate_salience`): compare feature and LLM systems against `gold_salient`.

## Immediate next implementation task
Implement `score_feature_salience` stage to populate:
1. `feature_salience_score`
2. `feature_salience_rank`
3. `feature_salience_label`

using `gold_salient` for calibration and evaluation.
