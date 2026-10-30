import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class TemporalAlertModel(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        conversation_dim=512,
        nhead=8,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.3,
        max_history_window=8,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.encoder.float()
        self.hidden_size = self.encoder.config.hidden_size
        self.conversation_dim = conversation_dim
        self.max_history_window = max_history_window

        self.turn_projection = nn.Linear(self.hidden_size, conversation_dim)
        self.position_embeddings = nn.Embedding(max_history_window, conversation_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conversation_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.conversation_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.alert_head = nn.Sequential(
            nn.Linear(conversation_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, history_lengths):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoder_outputs.last_hidden_state
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("Non-finite DeBERTa hidden states detected.")

        flat_turn_embeddings = hidden[:, 0, :]
        projected = self.turn_projection(flat_turn_embeddings)
        if not torch.isfinite(projected).all():
            raise FloatingPointError("Non-finite projected turn embeddings detected.")

        batch_size = int(history_lengths.numel())
        max_turns = int(history_lengths.max().item())
        dim = projected.shape[-1]
        device = projected.device
        sequence = projected.new_zeros((batch_size, max_turns, dim))
        padding_mask = torch.ones((batch_size, max_turns), dtype=torch.bool, device=device)

        cursor = 0
        for row_idx, length_tensor in enumerate(history_lengths):
            length = int(length_tensor.item())
            sequence[row_idx, :length] = projected[cursor:cursor + length]
            padding_mask[row_idx, :length] = False
            cursor += length

        positions = torch.arange(max_turns, device=device).unsqueeze(0).expand(batch_size, max_turns)
        sequence = sequence + self.position_embeddings(positions)
        contextual = self.conversation_transformer(sequence, src_key_padding_mask=padding_mask)
        if not torch.isfinite(contextual).all():
            raise FloatingPointError("Non-finite conversation Transformer outputs detected.")

        final_indices = (history_lengths - 1).long()
        current_context = contextual[torch.arange(batch_size, device=device), final_indices]
        logits = self.alert_head(current_context)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite Alert logits detected.")
        return logits, {
            "flat_turn_count": int(input_ids.shape[0]),
            "encoded_shape": tuple(input_ids.shape),
            "turn_embedding_shape": tuple(sequence.shape),
            "padding_mask_shape": tuple(padding_mask.shape),
            "logit_shape": tuple(logits.shape),
        }
