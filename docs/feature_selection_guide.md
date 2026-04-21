# Feature Selection & Optimization Guide

This document outlines the methodology and results for the feature engineering and selection process in the Squad RST V2 Salience Classifier.

## 1. Methodology: Permutation Importance
We used **Permutation Importance** to audit the hybrid model. This involves shuffling a single feature column in the validation set and measuring the drop in **Macro F1 score**.
- **High Importance**: Shuffling the feature causes a significant performance drop.
- **Zero Importance**: The model is ignoring the feature or has found a redundant "shortcut" (usually via the RoBERTa embeddings).

## 2. The "Backbone Drowning" Problem
Initially, most engineered features showed **0.0 importance**. This was caused by the RoBERTa backbone "cheating" by memorizing text patterns in the small training set (174 samples).

### Solutions Implemented:
1. **Frozen Backbone**: RoBERTa was frozen as a fixed feature extractor, forcing the model to find signals in the linguistic data.
2. **Gated Fusion / Strong MLP**: A high-capacity MLP was used to combine fixed [CLS] embeddings with engineered features.
3. **RST Meta-Mapping**: Specific RST relations (20+) were grouped into **9 high-density categories** to reduce categorical sparsity.

## 3. Final Optimized Feature Set
Based on the latest importance analysis, the following features have been selected for the production model:

### Primary Signals (Highest Importance)
- **RST Structural**: `rst_relation` (grouped), `span_importance_score`, `rst_tree_depth`.
- **Surprisal**: `surprisal_word_std`, `surprisal_word_max`.
- **Lexical/Semantic**: `contrast_marker_ratio`, `concreteness_ratio`.
- **Discourse**: `paragraph_discourse_continuity_score`, `prev_next_cohesion_score`.

### Pruned Features (Removed as Noise)
- `punctuation_pattern_comma_count`
- `concreteness_total`
- `avg_word_length`
- `surprisal_word_mean` (Redundant with `max` and `std`)

## 4. Feature Transformations
- **Log-Scaling**: Applied to `sentence_length_tokens`, `named_entity_count`, and `surprisal_word_max` to handle skewness.
- **Stratified Oversampling**: Minority salient classes are oversampled to a 50/50 distribution to stabilize gradients.

## 5. Performance Impact
- **Initial Recall**: 0% (Class collapse)
- **Intermediate (Unfrozen)**: 33% Recall
- **Optimized (Frozen + Grouped)**: **50% Recall** on salient sentence identification.
