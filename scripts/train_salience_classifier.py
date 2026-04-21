# Hybrid RoBERTa Classifier for Sentence Salience Detection
# Uses both contextual text and engineered tabular features
# Handles severe class imbalance via Focal Loss and class weights
# Hugging Face Trainer integration with proper metrics and early stopping

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import joblib

# =====================
# CONFIGURATION
# =====================
DATA_PATH = os.path.join('data', 'processed', 'salience_transformer_dataset.csv')
MODEL_NAME = 'roberta-base'
MAX_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 15  # Increased to give model time to refine boundaries
LEARNING_RATE = 1e-5 
WEIGHT_DECAY = 0.01
PATIENCE = 5 # Increased patience for refined tuning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================
# CUSTOM LOSS
# =====================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance.
    Down-weights easy examples and focuses on hard positives (salient class).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =====================
# METRICS COMPUTATION
# =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    
    # Calculate metrics, focusing on the minority/salient class (pos_label=1)
    p_salient, r_salient, f1_salient, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1, zero_division=0
    )
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'salient_precision': p_salient,
        'salient_recall': r_salient,
        'salient_f1': f1_salient
    }

# =====================
# MAIN FUNCTION
# =====================
def main():
    # 1. Load and prepare Data
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['gold_salient'].astype(int)

    for col in ['prev_sent_text', 'sent_text', 'next_sent_text']:
        if col not in df:
            df[col] = ''
    df['full_text'] = (
        df['prev_sent_text'].fillna('') + ' [SEP] ' +
        df['sent_text'].fillna('') + ' [SEP] ' +
        df['next_sent_text'].fillna('')
    )

    # 1. Feature Optimization: Prune noise while keeping core signals like rst_tree_depth
    num_cols = [
        'rst_tree_depth', 'span_importance_score', 'sentence_position_ratio',
        'prev_next_cohesion_score', 'paragraph_discourse_continuity_score',
        'sentence_length_tokens', 'syntactic_complexity_score', 'readability_score',
        'pos_ratio_NN', 'pos_ratio_NNP', 'pos_ratio_NNS', 'pos_ratio_VB',
        'pos_ratio_VBD', 'pos_ratio_VBG', 'pos_ratio_VBP', 'pos_ratio_VBZ', 'pos_ratio_JJ', 'pos_ratio_RB',
        'surprisal_word_std', 'surprisal_word_max', 'contrast_marker_ratio', 'causal_marker_ratio',
        'named_entity_count', 'concreteness_ratio'
    ]
    
    # 1.1 Apply Log-scaling to improve numerical stability for skewed features
    skewed_features = ['sentence_length_tokens', 'named_entity_count', 'surprisal_word_max']
    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            
    cat_cols = ['rst_relation', 'rst_nuclearity', 'cue_word_flags', 'prev_sent_label']

    df[num_cols] = df[num_cols].fillna(0)
    
    # 1.1 Group RST Relations to reduce sparsity
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

    # Stratified split to preserve class distribution across train/val/test
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=SEED, stratify=train_val_df['label'])

    splits = {
        'train': df.index.isin(train_df.index),
        'val': df.index.isin(val_df.index),
        'test': df.index.isin(test_df.index)
    }

    scaler = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler.fit(df.loc[splits['train'], num_cols])
    ohe.fit(df.loc[splits['train'], cat_cols])

    model_dir = os.path.join('models', 'hybrid_roberta')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(ohe, os.path.join(model_dir, 'ohe.joblib'))

    def prepare_features(subset):
        num = scaler.transform(subset[num_cols])
        cat = ohe.transform(subset[cat_cols])
        return np.hstack([num, cat])

    X_train = prepare_features(df[splits['train']])
    X_val = prepare_features(df[splits['val']])
    y_train = df.loc[splits['train'], 'label'].values
    y_val = df.loc[splits['val'], 'label'].values
    y_test = df.loc[splits['test'], 'label'].values

    # 2. Oversample the minority class in the training set
    # Instead of just weights, we physically duplicate the salient sentences
    # so the model is forced to learn their features.
    train_df_orig = df.loc[splits['train']].copy()
    salient_df = train_df_orig[train_df_orig['label'] == 1]
    non_salient_df = train_df_orig[train_df_orig['label'] == 0]
    
    # Calculate how many times to repeat salient samples to reach ~1:1 ratio
    multiplier = len(non_salient_df) // len(salient_df) if len(salient_df) > 0 else 1
    oversampled_salient = pd.concat([salient_df] * max(1, multiplier))
    
    # Final balanced training set
    train_df_balanced = pd.concat([non_salient_df, oversampled_salient]).sample(frac=1, random_state=SEED)
    
    texts_train = train_df_balanced['full_text'].tolist()
    texts_val = df.loc[splits['val'], 'full_text'].tolist()
    texts_test = df.loc[splits['test'], 'full_text'].tolist()
    y_train = train_df_balanced['label'].values
    
    # Re-transform oversampled features
    num_feats_train = scaler.transform(train_df_balanced[num_cols])
    cat_feats_train = ohe.transform(train_df_balanced[cat_cols])
    X_train_final = np.hstack([num_feats_train, cat_feats_train]).astype('float32')
    
    print(f"Oversampling complete. Original salient count: {len(salient_df)}, New: {len(oversampled_salient)}")
    print(f"Training on {len(train_df_balanced)} balanced samples.")
    feature_dim = X_train_final.shape[1]

    # Compute lighter class weights just for safety
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # 3. Setup Dataset
    from src.modeling.hybrid_dataset import HybridDataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prep validation/test features correctly
    num_feats_val = scaler.transform(df.loc[splits['val'], num_cols])
    cat_feats_val = ohe.transform(df.loc[splits['val'], cat_cols])
    X_val_final = np.hstack([num_feats_val, cat_feats_val]).astype('float32')
    
    num_feats_test = scaler.transform(df.loc[splits['test'], num_cols])
    cat_feats_test = ohe.transform(df.loc[splits['test'], cat_cols])
    X_test_final = np.hstack([num_feats_test, cat_feats_test]).astype('float32')

    train_ds = HybridDataset(texts_train, X_train_final, y_train, tokenizer, MAX_LEN)
    val_ds = HybridDataset(texts_val, X_val_final, y_val, tokenizer, MAX_LEN)
    test_ds = HybridDataset(texts_test, X_test_final, y_test, tokenizer, MAX_LEN)

    # 4. Setup Model
    from src.models.hybrid_roberta import HybridRoBERTa
    
    # Use standard CrossEntropy since we oversampled, but keep Focal with low gamma for refinement
    loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=1.0)
    
    model = HybridRoBERTa(feature_dim, model_name=MODEL_NAME, loss_fn=loss_fn)

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=1e-4, # Higher LR for training just the MLP head
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save model at the end of each epoch
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,    # Early stopping relies on this
        metric_for_best_model="macro_f1", # Prioritize macro F1 (handles imbalance)
        greater_is_better=True,
        seed=SEED,
        report_to="none",               # Change to 'wandb' or 'tensorboard' if needed
        remove_unused_columns=False     # Crucial: prevents Trainer from dropping 'labels' since it's not in forward() signature
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    # 6. Train the Model
    print("Starting Training...")
    trainer.train()

    # 7. Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    test_results = trainer.predict(test_ds)
    test_metrics = test_results.metrics
    
    # Trainer returns predictions in a tuple if outputs are complex, so check type
    logits = test_results.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    test_preds = np.argmax(logits, axis=-1)
    
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nTest Classification Report:")
    report = classification_report(y_test, test_preds, digits=4, output_dict=True)
    print(classification_report(y_test, test_preds, digits=4))

    # Save best model and tokenizer manually
    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
    trainer.save_model(os.path.join(model_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(model_dir, "best_model"))

    # Save report and metrics to interim folder
    import json
    interim_dir = os.path.join('data', 'interim')
    os.makedirs(interim_dir, exist_ok=True)
    report_path = os.path.join(interim_dir, 'salience_classifier_report.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({'classification_report': report, 'metrics': test_metrics}, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
