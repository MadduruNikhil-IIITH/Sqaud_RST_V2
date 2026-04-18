# Phase 1b: clean_data

## Goal
Validate sampled paragraphs for sentence extraction quality. Remove entries that would cause:
- Sentence segmentation failures
- Answer-to-sentence alignment errors
- Feature extraction issues

## Command
python -m src.pipeline.run_stage --stage clean_data

## Input
- data/interim/sample_manifest.json (from Phase 1)

## Outputs
- data/interim/cleaned_sample_manifest.json
- data/interim/cleaning_report.json

## Validation checks
1. **Paragraph length:** min 100 chars (ensures multiple sentences)
2. **Answer validity:** at least one Q-A pair with valid character offsets
3. **Encoding:** UTF-8 clean (no control chars except \n\t)
4. **QA structure:** questions non-empty, answers resolvable from context

## Success criteria
- Cleaned manifest has ≥ 12 paragraphs (most/all pass)
- Report shows removal reasons
- All retained paragraphs have valid sentence extraction potential
