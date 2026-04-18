"""
evaluate_inference.py

Evaluates Hybrid and LLM inference outputs against gold labels.
Saves evaluation metrics (accuracy, F1, classification report) as JSON.
"""
import pandas as pd
import json
import argparse
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_gold(gold_path):
    df = pd.read_csv(gold_path)
    # Accept either 'gold_salient' or 'label' as the gold column
    if 'gold_salient' in df.columns:
        return df[['para_id', 'sent_idx', 'gold_salient']].rename(columns={'gold_salient': 'label'})
    elif 'label' in df.columns:
        return df[['para_id', 'sent_idx', 'label']]
    else:
        raise ValueError('Gold file must contain gold_salient or label column')

def load_preds(pred_path, col):
    df = pd.read_csv(pred_path)
    return df[['para_id', 'sent_idx', col]]

def align_and_eval(gold_df, pred_df, pred_col):
    merged = pd.merge(gold_df, pred_df, on=['para_id', 'sent_idx'], how='inner', suffixes=('_gold', '_pred'))
    # Drop rows where prediction is NaN
    before = len(merged)
    merged = merged[merged[pred_col].notna()]
    after = len(merged)
    dropped = before - after
    y_true = merged['label']
    y_pred = merged[pred_col]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'classification_report': report,
        'n_samples': after,
        'n_dropped_nan_pred': dropped
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid and LLM inference outputs against gold labels.")
    parser.add_argument('--gold', required=True, help='Path to gold labels CSV')
    parser.add_argument('--hybrid', required=True, help='Path to Hybrid inference CSV')
    parser.add_argument('--llm', required=True, help='Path to LLM inference CSV')
    parser.add_argument('--eval-hybrid', required=True, help='Path to save Hybrid evaluation JSON')
    parser.add_argument('--eval-llm', required=True, help='Path to save LLM evaluation JSON')
    args = parser.parse_args()

    gold_df = load_gold(args.gold)
    hybrid_df = load_preds(args.hybrid, 'hybrid_salient')
    llm_df = load_preds(args.llm, 'llm_salient')

    hybrid_eval = align_and_eval(gold_df, hybrid_df, 'hybrid_salient')
    llm_eval = align_and_eval(gold_df, llm_df, 'llm_salient')

    with open(args.eval_hybrid, 'w', encoding='utf-8') as f:
        json.dump(hybrid_eval, f, indent=2, ensure_ascii=False)
    with open(args.eval_llm, 'w', encoding='utf-8') as f:
        json.dump(llm_eval, f, indent=2, ensure_ascii=False)
    print(f"Hybrid evaluation saved to {args.eval_hybrid}")
    print(f"LLM evaluation saved to {args.eval_llm}")

if __name__ == "__main__":
    main()
