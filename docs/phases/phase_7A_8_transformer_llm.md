# Phase 7A: Transformer Classifier Training

This phase trains a BERT-based classifier to predict sentence salience using the transformer dataset.

## Steps
1. Ensure `data/processed/salience_transformer_dataset.csv` is up to date.
2. Run:
   ```
   python scripts/train_bert_salience_classifier.py
   ```
3. Model and tokenizer are saved to `models/bert_salience_classifier/`.

# Phase 8: LLM-based Sentence Classification

This phase uses an LLM (e.g., OpenAI, HuggingFace) to classify sentences as salient or not.

## Steps
1. Ensure `data/processed/salience_transformer_dataset.csv` is up to date.
2. Run:
   ```
   python scripts/classify_sentences_with_llm.py
   ```
3. Output is saved to `data/processed/salience_transformer_dataset.csv`.

# Notes
- Both scripts use the same input dataset.
- Adjust model names and batch sizes as needed for your environment.
