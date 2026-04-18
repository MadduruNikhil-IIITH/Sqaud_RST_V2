# SQuAD RST Salience Study V2

## Quick start

1. Create env (preferred):
   - `conda env create -f environment.py312.yml`
   - `conda activate squad-rst-v2-py312`
2. Run env gate:
   - `python scripts/validate_environment_gate.py`
3. If gate fails for `isanlp_rst`, fallback:
   - `conda env create -f environment.py311.yml`
   - then `environment.py310.yml` if needed

### CUDA package note

- In this project, CUDA-backed PyTorch is installed via conda (`pytorch` + `pytorch-cuda=12.1`) from `pytorch` + `nvidia` channels.
- If you need a pure pip fallback, install torch from the PyTorch CUDA wheel index (not from separate `cuda-*` pip packages).
- `isanlp-rst` is the required RST package. A separate `isanlp` pip package is not needed and is not available on PyPI.

## Data input

Data is automatically downloaded from HuggingFace datasets (SQuAD v2.0). No manual download required.

## Stage-wise commands

Run from project root.

### Phase 0: environment gate

- `python scripts/validate_environment_gate.py`

### Phase 1: sample dataset

- `python -m src.pipeline.run_stage --stage sample_dataset`

### Phase 1b: clean data

- `python -m src.pipeline.run_stage --stage clean_data`

### Phase 2: segment sentences

- `python -m src.pipeline.run_stage --stage segment_sentences`

### Phase 3: build gold labels

- `python -m src.pipeline.run_stage --stage build_gold_labels`

### Phase 4: parse RST trees

- `python -m src.pipeline.run_stage --stage parse_rst_trees`

### Phase 5: feature extraction

Baseline (without PsychFormers execution):

- `python -m src.pipeline.run_stage --stage feature_extraction`

PsychFormers on GPU (validated working setup):

- `python -m src.pipeline.run_stage --stage feature_extraction --run-psychformers --psychformers-dir tools/PsychFormers --psychformers-model gpt2 --psychformers-decoder causal`

### Phase 6: score feature salience

- `python -m src.pipeline.run_stage --stage score_feature_salience`

Outputs:

- `data/processed/feature_table_scored.csv`
- `data/interim/feature_score_report.json`

### Transformer training export (prep)

- `python -m src.pipeline.run_stage --stage prepare_transformer_dataset`

Outputs:

- `data/processed/transformer_dataset.csv`
- `data/interim/transformer_split_report.json`

### Run full core pipeline (baseline)

- `python scripts/run_core_pipeline.py`

## Current implementation status

- Implemented: Phase 0 gate, Phase 1 sampling, Phase 1b cleaning, Phase 2 segmentation+offset checks, Phase 3 answer mapping+gold labels, Phase 4 rst artifact/image manifests, Phase 5 feature extraction, Phase 6 feature salience scoring.
- Next: add LLM salience annotation and evaluation stages.

## Phase 5 feature extraction

Default run:

- `python -m src.pipeline.run_stage --stage feature_extraction`

Optional PsychFormers-backed surprisal integration:

- `python -m src.pipeline.run_stage --stage feature_extraction --run-psychformers --psychformers-dir tools/PsychFormers --psychformers-model gpt2 --psychformers-decoder causal`

Main output:

- `data/processed/feature_table.csv`

PsychFormers intermediate outputs:

- `data/interim/psychformers/stims/`
- `data/interim/psychformers/output/`

## Phase docs and data model

- Phase reference docs: `docs/phases/`
- Canonical processed salience row model: `src/common/models.py` (`ProcessedSalienceRecord`)

### Phase 7A: Transformer classifier (BERT)

- Train BERT classifier:
  - `python scripts/train_bert_salience_classifier.py`
  - Model and tokenizer saved to `models/bert_salience_classifier/`

### Phase 8: LLM-based sentence classification

- Classify sentences with LLM (e.g., OpenAI, HuggingFace):
  - `python scripts/classify_sentences_with_llm.py`
  - Output: `data/processed/transformer_dataset_llm_scored.csv`

- Both scripts use `data/processed/transformer_dataset.csv` as input.
