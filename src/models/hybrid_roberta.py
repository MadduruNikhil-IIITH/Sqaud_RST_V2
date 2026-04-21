import torch
import torch.nn as nn
from transformers import AutoModel

class HybridRoBERTa(nn.Module):
    """
    Hybrid RoBERTa model with a FROZEN backbone.
    This treats RoBERTa as a fixed semantic feature extractor, forcing the 
    MLP head to find patterns in the engineered features.
    """
    def __init__(self, feature_dim, model_name="roberta-base", loss_fn=None):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Hard-Freeze the backbone
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # Dedicated branch for engineered features
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        combined_dim = self.roberta.config.hidden_size + 256
        self.norm = nn.LayerNorm(combined_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        self.loss_fn = loss_fn

    def forward(self, input_ids, attention_mask, engineered, labels=None, **kwargs):
        # 1. Extract Fixed Text Features
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :] # [B, 768]
        
        # 2. Process Engineered Features (Trainable)
        feat_emb = self.feature_encoder(engineered) # [B, 256]
        
        # 3. Fusion
        x = torch.cat([cls_emb, feat_emb], dim=1)
        x = self.norm(x)
        
        # 4. Classification
        logits = self.classifier(x)

        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)
            return (loss, logits)
        
        return logits
