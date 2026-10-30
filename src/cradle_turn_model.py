import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class FeedForwardHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class CradleTurnMultiTaskModel(nn.Module):
    """Current-turn CRADLE baseline with independent sigmoid/BCE heads."""

    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.alert_head = FeedForwardHead(hidden_size, 1, dropout)
        self.confirm_head = FeedForwardHead(hidden_size, 1, dropout)
        self.category_head = FeedForwardHead(hidden_size, 7, dropout)
        self.temporal_state_head = FeedForwardHead(hidden_size, 2, dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("Non-finite encoder last_hidden_state detected.")
        cls = hidden[:, 0, :]
        logits = {
            "alert": self.alert_head(cls),
            "confirm": self.confirm_head(cls),
            "category": self.category_head(cls),
            "temporal": self.temporal_state_head(cls),
        }
        for name, value in logits.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {name} logits detected.")
        return logits
