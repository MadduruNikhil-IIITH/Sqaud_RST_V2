import os
import torch
import pandas as pd
import numpy as np
import joblib
from transformers import AutoTokenizer
from src.models.hybrid_roberta import HybridRoBERTa
from src.modeling.hybrid_dataset import HybridDataset
from torch.utils.data import DataLoader

def debug_gate():
    MODEL_DIR = "models/hybrid_roberta"
    MODEL_NAME = "roberta-base"
    DATA_PATH = "data/inference/salience_transformer_dataset.csv"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Setup features exactly as in inference/importance
    num_cols = [
        'rst_tree_depth', 'span_importance_score', 'sentence_position_ratio',
        'prev_next_cohesion_score', 'paragraph_discourse_continuity_score',
        'sentence_length_tokens', 'syntactic_complexity_score', 'readability_score',
        'pos_ratio_NN', 'pos_ratio_NNP', 'pos_ratio_NNS', 'pos_ratio_VB',
        'pos_ratio_VBD', 'pos_ratio_VBG', 'pos_ratio_VBP', 'pos_ratio_VBZ', 'pos_ratio_JJ', 'pos_ratio_RB',
        'surprisal_word_std', 'surprisal_word_max', 'contrast_marker_ratio', 'causal_marker_ratio',
        'named_entity_count', 'concreteness_ratio'
    ]
    cat_cols = ['rst_relation', 'rst_nuclearity', 'cue_word_flags', 'prev_sent_label']
    
    # Apply Log-scaling
    skewed_features = ['sentence_length_tokens', 'named_entity_count', 'surprisal_word_max']
    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            
    df[cat_cols] = df[cat_cols].fillna('missing').astype(str)
    
    # 2. Load scalers and transform
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    ohe = joblib.load(os.path.join(MODEL_DIR, 'ohe.joblib'))
    
    num_all = scaler.transform(df[num_cols])
    cat_all = ohe.transform(df[cat_cols])
    
    # Split into RST and Other
    rst_num = num_all[:, :2]
    other_num = num_all[:, 2:]
    
    feature_names = ohe.get_feature_names_out(cat_cols)
    rst_cat_cols = ['rst_relation', 'rst_nuclearity']
    rst_ohe_mask = [any(name.startswith(c) for c in rst_cat_cols) for name in feature_names]
    rst_cat = cat_all[:, rst_ohe_mask]
    other_cat = cat_all[:, [not m for m in rst_ohe_mask]]
    
    X_rst = np.hstack([rst_num, rst_cat])
    X_other = np.hstack([other_num, other_cat])
    
    texts = (
        df['prev_sent_text'].fillna('') + ' [SEP] ' +
        df['sent_text'].fillna('') + ' [SEP] ' +
        df['next_sent_text'].fillna('')
    ).tolist()
    
    # 3. Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HybridRoBERTa(rst_dim=X_rst.shape[1], other_dim=X_other.shape[1], model_name=MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()
    
    # 4. Extract Gate Values
    # We need to hook the model or modify it to return the gate
    print("\n[DEBUG] Extracting Gate Values...")
    
    ds = HybridDataset(texts, X_rst, X_other, [0]*len(texts), tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            r_feats = batch['rst_feats'].to(DEVICE)
            o_feats = batch['other_feats'].to(DEVICE)
            
            # Re-implement the gate logic locally since we have the weights
            # gate = self.rst_gate_net(rst_feats)
            gate = model.rst_gate_net(r_feats)
            gate_val = gate.item()
            
            # Get logits for final prediction
            logits = model(input_ids, mask, r_feats, o_feats)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            
            results.append({
                'para_id': df.iloc[i]['para_id'],
                'sent_text': df.iloc[i]['sent_text'][:50],
                'gate_value': round(gate_val, 4),
                'salience_prob': round(prob, 4),
                'gold': df.iloc[i]['gold_salient']
            })

    res_df = pd.DataFrame(results)
    print("\n--- SAMPLE GATE VALUES ---")
    print(res_df.head(20).to_string(index=False))
    
    print("\n--- GATE STATISTICS ---")
    print(res_df['gate_value'].describe())
    
    # Check if gate correlates with gold
    print("\n--- GATE VS GOLD SALIENCE ---")
    print(res_df.groupby('gold')['gate_value'].mean())

if __name__ == "__main__":
    debug_gate()
