import torch
import torch.nn as nn
from transformers import AutoModel

class HybridRoBERTa(nn.Module):
    """
    Hybrid RoBERTa model V3: RST-Driven Gated Attention.
    Uses the RST features (discourse structure) as a master gate to modulate
    how much the model trusts the semantic text vs. other linguistic signals.
    """
    def __init__(self, rst_dim, other_dim, model_name="roberta-base", loss_fn=None):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # Branch 1: The RST "Discourse Controller"
        self.rst_gate_net = nn.Sequential(
            nn.Linear(rst_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # This is the salience multiplier based on RST alone
        )
        
        # Branch 2: Other Linguistic Features (Surprisal, POS, etc.)
        self.linguistic_encoder = nn.Sequential(
            nn.Linear(other_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Branch 3: Semantic Projector
        self.semantic_projection = nn.Linear(self.roberta.config.hidden_size, 256)
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        self.loss_fn = loss_fn

    def forward(self, input_ids, attention_mask, rst_feats, other_feats, labels=None, **kwargs):
        # 1. Get Discourse Gate from RST
        # This weight [0, 1] determines the "Structural Importance"
        discourse_weight = self.rst_gate_net(rst_feats) # [B, 1]
        
        # 2. Extract Semantic Features
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :] # [B, 768]
        sem_feat = self.semantic_projection(cls_emb) # [B, 256]
        
        # 3. Extract Other Linguistic Features
        ling_feat = self.linguistic_encoder(other_feats) # [B, 256]
        
        # 4. Gated Fusion: Discourse weight modulates the entire combined feature space
        combined = torch.cat([sem_feat, ling_feat], dim=1)
        
        # THE CORE IMPROVEMENT: The RST gate scales the information flow
        gated_fused = combined * discourse_weight
        
        # 5. Classification
        logits = self.classifier(gated_fused)

        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)
            return (loss, logits)
        
        return logits
