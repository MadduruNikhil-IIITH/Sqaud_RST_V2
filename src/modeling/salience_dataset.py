
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import re
from pathlib import Path

class SalienceDataset(Dataset):
    """
    Generic dataset for sentence/paragraph classification tasks.
    Can be used for both salience and LLM classifier training/inference.
    Optionally retrieves and cleans paragraph context from a manifest file using para_id.
    """
    def __init__(self, csv_path, tokenizer, text_col='context_window_text', label_col='gold_salient', max_length=256, manifest_path=None, use_paragraph_context=False):
        df = pd.read_csv(csv_path)
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist() if label_col in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.para_ids = df['para_id'].astype(str).tolist() if 'para_id' in df.columns else None
        self.use_paragraph_context = use_paragraph_context and manifest_path is not None
        self.paragraph_map = {}
        if self.use_paragraph_context:
            self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        for para in manifest.get('paragraphs', []):
            cleaned = self._clean_paragraph(para.get('context', ''))
            self.paragraph_map[para['para_id']] = cleaned

    def _clean_paragraph(self, text):
        # Remove extra whitespace, normalize quotes, etc.
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('“', '"').replace('”', '"').replace("’", "'")
        return text.strip()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.use_paragraph_context and self.para_ids:
            para_id = self.para_ids[idx]
            para_text = self.paragraph_map.get(para_id, '')
            if para_text:
                text = para_text + ' ' + text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
