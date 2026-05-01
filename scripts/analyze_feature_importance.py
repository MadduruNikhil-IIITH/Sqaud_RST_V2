import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from src.models.hybrid_roberta import HybridRoBERTa
from src.modeling.hybrid_dataset import HybridDataset
from torch.utils.data import DataLoader

def analyze_importance():
    MODEL_DIR = "models/hybrid_roberta"
    MODEL_NAME = "roberta-base"
    DATA_PATH = "data/processed/salience_transformer_dataset.csv"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['gold_salient'].astype(int)
    
    # Define columns exactly as in training
    num_cols = [
        'rst_tree_depth', 'span_importance_score', 'sentence_position_ratio',
        'prev_next_cohesion_score', 'paragraph_discourse_continuity_score',
        'sentence_length_tokens', 'syntactic_complexity_score', 'readability_score',
        'pos_ratio_NN', 'pos_ratio_NNP', 'pos_ratio_NNS', 'pos_ratio_VB',
        'pos_ratio_VBD', 'pos_ratio_VBG', 'pos_ratio_VBP', 'pos_ratio_VBZ', 'pos_ratio_JJ', 'pos_ratio_RB',
        'surprisal_word_std', 'surprisal_word_max', 'contrast_marker_ratio', 'causal_marker_ratio',
        'named_entity_count', 'concreteness_ratio'
    ]
    
    # Apply Log-scaling (Must match training exactly)
    skewed_features = ['sentence_length_tokens', 'named_entity_count', 'surprisal_word_max']
    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            
    cat_cols = ['rst_relation', 'rst_nuclearity', 'cue_word_flags', 'prev_sent_label']
    
    # Fill missing values
    df[num_cols] = df[num_cols].fillna(0)
    
    # RST Grouping (Must match training/inference exactly)
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
    
    # Use validation split
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    if len(val_df) == 0:
        print("No validation split found in CSV. Using full dataset for analysis.")
        val_df = df
    
    texts = (
        val_df['prev_sent_text'].fillna('') + ' [SEP] ' +
        val_df['sent_text'].fillna('') + ' [SEP] ' +
        val_df['next_sent_text'].fillna('')
    ).tolist()
    
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    ohe = joblib.load(os.path.join(MODEL_DIR, 'ohe.joblib'))
    
    # RST Indices
    rst_num_cols = ['rst_tree_depth', 'span_importance_score']
    rst_cat_cols = ['rst_relation', 'rst_nuclearity']
    
    def split_features(subset_df, sc, oh):
        num_all = sc.transform(subset_df[num_cols])
        cat_all = oh.transform(subset_df[cat_cols])
        
        rst_num = num_all[:, :2]
        other_num = num_all[:, 2:]
        
        feature_names = oh.get_feature_names_out(cat_cols)
        rst_ohe_mask = [any(name.startswith(c) for c in rst_cat_cols) for name in feature_names]
        rst_cat = cat_all[:, rst_ohe_mask]
        other_cat = cat_all[:, [not m for m in rst_ohe_mask]]
        
        return np.hstack([rst_num, rst_cat]), np.hstack([other_num, other_cat])

    X_rst_base, X_other_base = split_features(val_df, scaler, ohe)
    
    rst_dim = X_rst_base.shape[1]
    other_dim = X_other_base.shape[1]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HybridRoBERTa(rst_dim=rst_dim, other_dim=other_dim, model_name=MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    def get_f1(r_feats, o_feats):
        ds = HybridDataset(texts, r_feats, o_feats, labels, tokenizer)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                r_batch = batch['rst_feats'].to(DEVICE)
                o_batch = batch['other_feats'].to(DEVICE)
                logits = model(input_ids, attention_mask, r_batch, o_batch)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return f1_score(labels, preds, average='macro')

    print("Calculating baseline Macro F1...")
    baseline_f1 = get_f1(X_rst_base, X_other_base)
    print(f"Baseline Macro F1: {baseline_f1:.4f}")

    importance_results = []

    # Analyze Numerical Features
    for col in tqdm(num_cols, desc="Analyzing Numerical Features"):
        temp_df = val_df.copy()
        temp_df[col] = np.random.permutation(temp_df[col].values)
        r_p, o_p = split_features(temp_df, scaler, ohe)
        
        perm_f1 = get_f1(r_p, o_p)
        importance = baseline_f1 - perm_f1
        importance_results.append({'feature': col, 'importance': importance})

    # Analyze Categorical Features
    for col in tqdm(cat_cols, desc="Analyzing Categorical Features"):
        temp_df = val_df.copy()
        temp_df[col] = np.random.permutation(temp_df[col].values)
        r_p, o_p = split_features(temp_df, scaler, ohe)
        
        perm_f1 = get_f1(r_p, o_p)
        importance = baseline_f1 - perm_f1
        importance_results.append({'feature': col, 'importance': importance})

    importance_df = pd.DataFrame(importance_results).sort_values(by='importance', ascending=False)
    print("\n--- FEATURE IMPORTANCE (Impact on Macro F1) ---")
    print(importance_df.head(20))
    
    importance_df.to_csv("data/interim/feature_importance.csv", index=False)
    print("\nResults saved to data/interim/feature_importance.csv")

if __name__ == "__main__":
    analyze_importance()
