# SQuAD v2 Question Generation (QG) Pipeline Process

## Overview
This pipeline generates questions for SQuAD v2-style data using multiple modes and state-of-the-art models. It is modular, reproducible, and designed for efficient use on consumer GPUs (e.g., RTX 4060 8GB, Colab T4).

## Modes
- **Hybrid Salient:** Uses salient sentences from hybrid_salient column + full paragraph
- **LLM Salient:** Uses salient sentences from llm_salient column + full paragraph
- **Zero-shot:** Uses only the full paragraph (no salience guidance)

## Input Files (from `data/inference/`)
- `cleaned_sample_manifest.json`
- `sentence_table_llm_inference.csv`
- `feature_table_hybrid_inference.csv`

## Output Files (in `data/qg/outputs/`)
- `qg_hybrid_salient_generated.jsonl`
- `qg_llm_salient_generated.jsonl`
- `qg_zero_shot_generated.jsonl`
Each line contains `{ "input": ..., "output": ... }` for full traceability.

## Source Structure
- `src/qg/model_loader.py` — Model loading abstraction
- `src/qg/qg_inference.py` — Batched inference logic
- `scripts/run_qg_pipeline.py` — Main wrapper script
- `docs/qg_pipeline_process.md` — This documentation


## Model Recommendation & Comparison

| Model Name                        | VRAM/Speed | QG Quality | Structure | Notes                                  |
|------------------------------------|------------|------------|----------|----------------------------------------|
| valhalla/t5-base-e2e-qg           | <2GB       | High       | High     | **Best for SQuAD-style QG, fast, robust, structured** |
| Qwen/Qwen1.5-0.5B                 | <2GB       | High       | High     | Efficient, long context, chat/instruct variants      |
| Qwen/Qwen1.5-1.8B                 | ~4GB       | High       | High     | Efficient, long context, chat/instruct variants      |
| Qwen/Qwen1.5-7B-Chat              | 8GB+       | Very High  | High     | Large context, strong LLM, quantized for 8GB VRAM    |
| microsoft/Phi-4-mini-instruct     | <2GB       | Medium     | Medium   | Efficient, more open-ended, less structured QG       |
| google/t5-base-qg-prepend         | <2GB       | High       | High     | Similar to above, prepend-style input                |
| allenai/unifiedqa-t5-base         | <2GB       | High       | High     | QA-tuned, may generalize better                     |
| Meta-Llama-3.1-8B-Instruct (4bit) | 8GB+       | Very High  | Medium   | Slower, more setup, optional                        |

**Best model:** `valhalla/t5-base-e2e-qg` (best structure, speed, and SQuAD-style QG quality)

**Qwen models** are also recommended for long context and strong general LLM performance, especially Qwen1.5-0.5B and Qwen1.5-1.8B for 8GB VRAM or less.

**Phi models** are efficient and open-ended but less structured for SQuAD-style QG.

## How to Run
```bash
python scripts/run_qg_pipeline.py --model valhalla/t5-base-e2e-qg
```

## Extensibility
- Add new models by updating `SUPPORTED_MODELS` in `model_loader.py`.
- Add new modes by extending input construction logic in `qg_inference.py`.

## Evaluation
- Not included in this implementation. Will be handled as a separate phase.

## Notes
- All inputs and outputs are paired and saved for reproducibility.
- Designed for easy experimentation and extension.
