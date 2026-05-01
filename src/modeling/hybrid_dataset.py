import torch
from torch.utils.data import Dataset

class HybridDataset(Dataset):
    """
    Custom Dataset for hybrid RoBERTa + engineered features.
    Separates RST features from other linguistic features for gated fusion.
    """
    def __init__(self, texts, rst_feats, other_feats, labels, tokenizer, max_len=256):
        self.texts = texts
        self.rst_feats = torch.tensor(rst_feats, dtype=torch.float32)
        self.other_feats = torch.tensor(other_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'rst_feats': self.rst_feats[idx],
            'other_feats': self.other_feats[idx],
            'labels': self.labels[idx]
        }
