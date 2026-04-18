"""
hybrid_inference.py

Module for running Hybrid RoBERTa-based salience inference on a feature table CSV.
Loads pre-trained model weights from the models directory.
"""
import torch
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
from src.models.hybrid_roberta import HybridRoBERTa
from src.modeling.hybrid_dataset import HybridDataset
from torch.utils.data import DataLoader
import joblib
from tqdm import tqdm
import numpy as np

def run_hybrid_inference(feature_table_csv: Path, output_csv: Path, model_path: str = "models/hybrid_roberta/best_model.pt"):
    df = pd.read_csv(feature_table_csv)
    # Remove 'split' and 'gold_salient' columns from input before feature extraction
    for col in ["split", "gold_salient"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Match training feature processing ---
    # Define feature columns (must match training)
    num_cols = [
        'avg_word_length', 'sentence_length_words', 'type_token_ratio',
        'causal_marker_ratio', 'contrast_marker_ratio', 'named_entity_density',
        'rst_tree_depth', 'span_importance_score', 'sentence_position_ratio',
        'named_entity_count', 'prev_next_cohesion_score', 'paragraph_discourse_continuity_score',
        'content_word_density', 'sentence_length_tokens', 'lexical_density',
        'syntactic_complexity_score', 'readability_score',
        'pos_ratio_NN', 'pos_ratio_NNP', 'pos_ratio_NNS', 'pos_ratio_VB',
        'pos_ratio_VBD', 'pos_ratio_VBG', 'pos_ratio_VBN', 'pos_ratio_VBP',
        'pos_ratio_VBZ', 'pos_ratio_JJ', 'pos_ratio_RB',
        'punctuation_pattern_comma_count', 'punctuation_pattern_semicolon_count',
        'concreteness_noun_count', 'concreteness_total', 'concreteness_ratio',
        'surprisal_sentence_total', 'surprisal_sentence_per_token',
        'surprisal_word_mean', 'surprisal_word_max', 'surprisal_word_std'
    ]
    cat_cols = ['rst_relation', 'rst_nuclearity', 'cue_word_flags', 'prev_sent_label']

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('missing').astype(str)

    # Aggregate text context as in training
    for col in ['prev_sent_text', 'sent_text', 'next_sent_text']:
        if col not in df:
            df[col] = ''
    df['full_text'] = (
        df['prev_sent_text'].fillna('') + ' [SEP] ' +
        df['sent_text'].fillna('') + ' [SEP] ' +
        df['next_sent_text'].fillna('')
    )

    # Load scaler and encoder (must be saved during training)
    scaler = joblib.load('models/hybrid_roberta/scaler.joblib')
    ohe = joblib.load('models/hybrid_roberta/ohe.joblib')

    # Sequential inference: update prev_sent_label for each sentence
    from copy import deepcopy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid Inference] Using device: {device}")
    model = HybridRoBERTa(feature_dim=scaler.transform(df[num_cols]).shape[1] + ohe.transform(df[cat_cols]).shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    all_preds = []
    df = df.sort_values(["para_id", "sent_idx"]).reset_index(drop=True)
    prev_label_map = {}  # para_id -> prev_sent_label
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Hybrid Sequential Inference"):
        para_id = row["para_id"] if "para_id" in row else 0
        # Set prev_sent_label for this row
        if para_id not in prev_label_map:
            prev_sent_label = 'missing'
        else:
            prev_sent_label = str(prev_label_map[para_id])
        row_cpy = deepcopy(row)
        row_cpy["prev_sent_label"] = prev_sent_label
        # Prepare features for this row using DataFrame to preserve column names
        num_df = pd.DataFrame([row_cpy[num_cols].values], columns=num_cols)
        cat_df = pd.DataFrame([[row_cpy[c] for c in cat_cols]], columns=cat_cols)
        num = scaler.transform(num_df)
        cat = ohe.transform(cat_df)
        feats = np.hstack([num, cat]).astype('float32')
        text = row_cpy['full_text']
        labels = [0]  # Dummy label
        dataset = HybridDataset([text], feats, labels, tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                engineered = batch["engineered"].to(device, non_blocking=True)
                logits = model(input_ids, attention_mask, engineered)
                pred = torch.argmax(logits, dim=1).detach().cpu().item()
                all_preds.append(pred)
                prev_label_map[para_id] = pred
    df["hybrid_salient"] = all_preds
    # Remove 'split' and 'gold_salient' columns if present before saving
    for col in ["split", "gold_salient"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df.to_csv(output_csv, index=False)
    print(f"Hybrid RoBERTa-based salience tagging complete. Output saved to {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Hybrid RoBERTa-based salience inference on a feature table CSV.")
    parser.add_argument("feature_table_csv", type=str, help="Path to the input transformer dataset CSV")
    parser.add_argument("output_csv", type=str, help="Path to save the output CSV with hybrid_salient column")
    parser.add_argument("--model-path", type=str, default="models/hybrid_roberta/best_model.pt", help="Path to the model weights")
    args = parser.parse_args()
    run_hybrid_inference(Path(args.feature_table_csv), Path(args.output_csv), model_path=args.model_path)
