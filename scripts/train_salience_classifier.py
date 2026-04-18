# Hybrid RoBERTa Classifier for Sentence Salience Detection
# Uses both contextual text and engineered tabular features
# Handles missing values, scaling, and categorical encoding
# Modular, scalable, and ready for larger datasets

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter
from tqdm import tqdm
import joblib

# =====================
# CONFIGURATION
# =====================
DATA_PATH = os.path.join('data', 'processed', 'salience_transformer_dataset.csv')
MODEL_NAME = 'roberta-base'
MAX_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================

# DATA LOADING & PREPROCESSING
# =====================
from sklearn.model_selection import train_test_split
df = pd.read_csv(DATA_PATH)
df['label'] = df['gold_salient'].astype(int)

# Rich text context
for col in ['prev_sent_text', 'sent_text', 'next_sent_text']:
	if col not in df:
		df[col] = ''
df['full_text'] = (
	df['prev_sent_text'].fillna('') + ' [SEP] ' +
	df['sent_text'].fillna('') + ' [SEP] ' +
	df['next_sent_text'].fillna('')
)

# Feature columns
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

# Stratified random split for balanced classes
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=SEED, stratify=train_val_df['label'])

# Create split masks
splits = {
	'train': df.index.isin(train_df.index),
	'val': df.index.isin(val_df.index),
	'test': df.index.isin(test_df.index)
}

# Fit scaler and encoder on train only
scaler = StandardScaler()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler.fit(df.loc[splits['train'], num_cols])
ohe.fit(df.loc[splits['train'], cat_cols])

# Save scaler and encoder for inference
joblib.dump(scaler, 'models/hybrid_roberta/scaler.joblib')
joblib.dump(ohe, 'models/hybrid_roberta/ohe.joblib')

def prepare_features(subset):
	num = scaler.transform(subset[num_cols])
	cat = ohe.transform(subset[cat_cols])
	return np.hstack([num, cat])

# Prepare features and labels
X_train = prepare_features(df[splits['train']])
X_val = prepare_features(df[splits['val']])
X_test = prepare_features(df[splits['test']])

y_train = df.loc[splits['train'], 'label'].values
y_val = df.loc[splits['val'], 'label'].values
y_test = df.loc[splits['test'], 'label'].values

texts_train = df.loc[splits['train'], 'full_text'].tolist()
texts_val = df.loc[splits['val'], 'full_text'].tolist()
texts_test = df.loc[splits['test'], 'full_text'].tolist()

feature_dim = X_train.shape[1]

# =====================
# DATASET
# =====================
from src.modeling.hybrid_dataset import HybridDataset

# =====================
# MODEL
# =====================
from src.models.hybrid_roberta import HybridRoBERTa

# =====================
# TRAINING UTILS
# =====================
def get_class_weights(labels):
	counts = Counter(labels)
	total = sum(counts.values())
	weights = [total / counts[i] for i in range(2)]
	weights = torch.tensor(weights, dtype=torch.float32)
	return weights

def train_epoch(model, loader, optimizer, criterion):
	model.train()
	total_loss = 0
	all_preds, all_labels = [], []
	for batch in tqdm(loader, desc='Training', leave=False):
		optimizer.zero_grad()
		input_ids = batch['input_ids'].to(DEVICE)
		attention_mask = batch['attention_mask'].to(DEVICE)
		engineered = batch['engineered'].to(DEVICE)
		labels = batch['labels'].to(DEVICE)
		logits = model(input_ids, attention_mask, engineered)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
		preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
		all_preds.extend(preds)
		all_labels.extend(labels.cpu().numpy())
	f1 = f1_score(all_labels, all_preds, average='macro')
	acc = accuracy_score(all_labels, all_preds)
	return total_loss / len(loader), f1, acc

def eval_epoch(model, loader, criterion):
	model.eval()
	total_loss = 0
	all_preds, all_labels = [], []
	with torch.no_grad():
		for batch in tqdm(loader, desc='Evaluating', leave=False):
			input_ids = batch['input_ids'].to(DEVICE)
			attention_mask = batch['attention_mask'].to(DEVICE)
			engineered = batch['engineered'].to(DEVICE)
			labels = batch['labels'].to(DEVICE)
			logits = model(input_ids, attention_mask, engineered)
			loss = criterion(logits, labels)
			total_loss += loss.item()
			preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
			all_preds.extend(preds)
			all_labels.extend(labels.cpu().numpy())
	f1 = f1_score(all_labels, all_preds, average='macro')
	acc = accuracy_score(all_labels, all_preds)
	return total_loss / len(loader), f1, acc, all_preds, all_labels

# =====================
# MAIN TRAINING LOOP
# =====================
def main():
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	train_ds = HybridDataset(texts_train, X_train, y_train, tokenizer, MAX_LEN)
	val_ds = HybridDataset(texts_val, X_val, y_val, tokenizer, MAX_LEN)
	test_ds = HybridDataset(texts_test, X_test, y_test, tokenizer, MAX_LEN)

	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

	model = HybridRoBERTa(feature_dim, model_name=MODEL_NAME).to(DEVICE)
	class_weights = get_class_weights(y_train).to(DEVICE)
	criterion = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

	best_f1 = 0
	model_dir = os.path.join('models', 'hybrid_roberta')
	os.makedirs(model_dir, exist_ok=True)
	best_model_path = os.path.join(model_dir, 'best_model.pt')
	for epoch in range(NUM_EPOCHS):
		train_loss, train_f1, train_acc = train_epoch(model, train_loader, optimizer, criterion)
		val_loss, val_f1, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
		print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f}")
		if val_f1 > best_f1:
			best_f1 = val_f1
			torch.save(model.state_dict(), best_model_path)

	# Load best model and evaluate on test
	model.load_state_dict(torch.load(best_model_path))
	test_loss, test_f1, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
	print(f"\nTest Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")
	print("\nTest Classification Report:")
	report = classification_report(test_labels, test_preds, digits=4, output_dict=True)
	print(classification_report(test_labels, test_preds, digits=4))

	# Save report and metrics to interim folder
	import json
	interim_dir = os.path.join('data', 'interim')
	os.makedirs(interim_dir, exist_ok=True)
	report_path = os.path.join(interim_dir, 'salience_classifier_report.json')
	metrics = {
		'test_loss': test_loss,
		'test_f1': test_f1,
		'test_acc': test_acc
	}
	with open(report_path, 'w', encoding='utf-8') as f:
		json.dump({'classification_report': report, 'metrics': metrics}, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
	main()
