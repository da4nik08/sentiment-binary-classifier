import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
from transformers import AutoTokenizer, AutoModel


class MiniLMSentimentClassifier(nn.Module):
    def __init__(self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dropout_p: float = 0.2,
        device: str = "cuda"):
        super().__init__()

        self.device = device
        self.encoder = AutoModel.from_pretrained(model_name)

        embedding_dim = self.encoder.config.hidden_size  # 384 for MiniLM
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(embedding_dim, 1)
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1]:   # Xavier для нових Linear-шарів
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def get_info(self):
        return 'all-MiniLM-L6-v2', 'Fine-tuning'

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, token_type_ids, attention_mask) -> torch.Tensor:
        model_output = self.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

        # mean pooling з attention_mask
        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden_size)
        sentence_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        sentence_embeddings = self.dropout(sentence_embeddings)
        return self.fc1(sentence_embeddings)