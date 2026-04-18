import torch
import torch.nn as nn
from transformers import AutoModel

class HybridRoBERTa(nn.Module):
    """
    Hybrid RoBERTa model that concatenates [CLS] embedding with engineered features
    and passes through an MLP classifier head.
    """
    def __init__(self, feature_dim, model_name="roberta-base"):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        for p in self.roberta.parameters():
            p.requires_grad = False  # Freeze backbone initially

        self.norm = nn.LayerNorm(self.roberta.config.hidden_size + feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, engineered):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        x = torch.cat([cls, engineered], dim=1)
        x = self.norm(x)
        return self.classifier(x)
