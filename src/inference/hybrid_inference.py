"""
hybrid_inference.py

Module for running Hybrid RoBERTa-based salience inference on a feature table CSV.
V2: Uses Top-K probability ranking instead of binary argmax classification.
Guarantees K sentences selected per paragraph (no more "total miss" failures).
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

# Number of sentences to select per paragraph
TOP_K = 2


def run_hybrid_inference(feature_table_csv: Path, output_csv: Path, model_path: str = "models/hybrid_roberta/best_model.pt"):
    df = pd.read_csv(feature_table_csv)
    # Remove 'split' and 'gold_salient' columns from input before feature extraction
    for col in ["split", "gold_salient"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Match training feature processing ---
    # Define feature columns (must match training)
    num_cols = [
        'rst_tree_depth', 'span_importance_score', 'sentence_position_ratio',
        'prev_next_cohesion_score', 'paragraph_discourse_continuity_score',
        'sentence_length_tokens', 'syntactic_complexity_score', 'readability_score',
        'pos_ratio_NN', 'pos_ratio_NNP', 'pos_ratio_NNS', 'pos_ratio_VB',
        'pos_ratio_VBD', 'pos_ratio_VBG', 'pos_ratio_VBP', 'pos_ratio_VBZ', 'pos_ratio_JJ', 'pos_ratio_RB',
        'surprisal_word_std', 'surprisal_word_max', 'contrast_marker_ratio', 'causal_marker_ratio',
        'named_entity_count', 'concreteness_ratio'
    ]
    
    # 1.1 Apply Log-scaling (Must match training)
    skewed_features = ['sentence_length_tokens', 'named_entity_count', 'surprisal_word_max']
    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            
    cat_cols = ['rst_relation', 'rst_nuclearity', 'cue_word_flags', 'prev_sent_label']

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(0)
    
    # RST Grouping (Must match training exactly)
    rst_mapping = {
        'attribution': 'Attribution', 'background': 'Background',
        'cause': 'Causal', 'result': 'Causal', 'consequence': 'Causal',
        'comparison': 'Contrast', 'contrast': 'Contrast', 'concession': 'Contrast', 'adversative': 'Contrast',
        'elaboration': 'Elaboration', 'explanation': 'Elaboration', 'evidence': 'Elaboration',
        'temporal': 'Temporal', 'sequence': 'Temporal', 'circumstance': 'Temporal',
        'joint': 'Joint', 'same-unit': 'Joint',
        'restatement': 'Restatement', 'context': 'Context', 'purpose': 'Purpose'
    }
    def group_rst(rel):
        rel = str(rel).lower()
        for key, val in rst_mapping.items():
            if key in rel: return val
        return 'Other'

    df['rst_relation'] = df['rst_relation'].apply(group_rst)
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
    model_dir = Path(model_path).parent
    scaler = joblib.load(model_dir / 'scaler.joblib')
    ohe = joblib.load(model_dir / 'ohe.joblib')

    # Sequential inference: update prev_sent_label for each sentence
    # RST Indices (Must match training splitting exactly)
    rst_num_cols = ['rst_tree_depth', 'span_importance_score']
    rst_cat_cols = ['rst_relation', 'rst_nuclearity']

    # Sequential inference: update prev_sent_label for each sentence
    from copy import deepcopy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid Inference] Using device: {device}")
    
    # Calculate dimensions
    dummy_num = scaler.transform(df.head(1)[num_cols])
    dummy_cat = ohe.transform(df.head(1)[cat_cols])
    
    # RST dimensions
    rst_num_dim = 2 # rst_tree_depth, span_importance_score
    feature_names = ohe.get_feature_names_out(cat_cols)
    rst_ohe_mask = [any(name.startswith(c) for c in rst_cat_cols) for name in feature_names]
    rst_cat_dim = sum(rst_ohe_mask)
    
    rst_dim = rst_num_dim + rst_cat_dim
    other_dim = (dummy_num.shape[1] - rst_num_dim) + (dummy_cat.shape[1] - rst_cat_dim)
    
    model = HybridRoBERTa(rst_dim=rst_dim, other_dim=other_dim)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Store probabilities instead of hard predictions
    all_probs = []
    df = df.sort_values(["para_id", "sent_idx"]).reset_index(drop=True)
    prev_label_map = {}  # para_id -> prev_sent_label
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Hybrid Sequential Inference"):
        para_id = row["para_id"] if "para_id" in row else 0
        if para_id not in prev_label_map:
            prev_sent_label = 'missing'
        else:
            prev_sent_label = str(prev_label_map[para_id])
        
        row_cpy = deepcopy(row)
        row_cpy["prev_sent_label"] = prev_sent_label
        
        # Prepare features
        num_df = pd.DataFrame([row_cpy[num_cols].values], columns=num_cols)
        cat_df = pd.DataFrame([[row_cpy[c] for c in cat_cols]], columns=cat_cols)
        num_all = scaler.transform(num_df)
        cat_all = ohe.transform(cat_df)
        
        # Split features
        rst_feats = np.hstack([num_all[:, :2], cat_all[:, rst_ohe_mask]]).astype('float32')
        other_feats = np.hstack([num_all[:, 2:], cat_all[:, [not m for m in rst_ohe_mask]]]).astype('float32')
        
        text = row_cpy['full_text']
        labels = [0]  # Dummy label
        dataset = HybridDataset([text], rst_feats, other_feats, labels, tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                rst_batch = batch["rst_feats"].to(device)
                other_batch = batch["other_feats"].to(device)
                logits = model(input_ids, attention_mask, rst_batch, other_batch)
                
                probs = torch.softmax(logits, dim=1)
                salient_prob = probs[0, 1].detach().cpu().item()
                all_probs.append(salient_prob)
                
                pred = torch.argmax(logits, dim=1).detach().cpu().item()
                prev_label_map[para_id] = pred

    df["hybrid_prob"] = all_probs

    # V2: Top-K selection per paragraph (guarantees K sentences per paragraph)
    df["hybrid_salient"] = 0
    for para_id in df["para_id"].unique():
        mask = df["para_id"] == para_id
        para_probs = df.loc[mask, "hybrid_prob"]
        n_sentences = len(para_probs)
        # Select top-K, but don't exceed the number of sentences
        k = min(TOP_K, n_sentences)
        top_k_idx = para_probs.nlargest(k).index
        df.loc[top_k_idx, "hybrid_salient"] = 1

    # Report selection stats
    pos_rate = df["hybrid_salient"].mean()
    per_para = df.groupby("para_id")["hybrid_salient"].sum()
    print(f"\n[Hybrid Inference V2] Top-K={TOP_K} selection results:")
    print(f"  Positive rate: {pos_rate:.1%} ({df['hybrid_salient'].sum()}/{len(df)} sentences)")
    print(f"  Per-paragraph selections: {dict(per_para)}")
    print(f"  Probability range: [{df['hybrid_prob'].min():.4f}, {df['hybrid_prob'].max():.4f}]")

    # Remove 'split' and 'gold_salient' columns if present before saving
    for col in ["split", "gold_salient"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df.to_csv(output_csv, index=False)
    print(f"Hybrid RoBERTa Top-K salience tagging complete. Output saved to {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Hybrid RoBERTa-based salience inference on a feature table CSV.")
    parser.add_argument("feature_table_csv", type=str, help="Path to the input transformer dataset CSV")
    parser.add_argument("output_csv", type=str, help="Path to save the output CSV with hybrid_salient column")
    parser.add_argument("--model-path", type=str, default="models/hybrid_roberta/best_model.pt", help="Path to the model weights")
    args = parser.parse_args()
    run_hybrid_inference(Path(args.feature_table_csv), Path(args.output_csv), model_path=args.model_path)
