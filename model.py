

from typing import List, Optional

import torch
from torch import nn


class BERTEncoder(nn.Module):
    '''
    Taken from https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html with 
    minimal modification
    '''
    def __init__(
        self,
        vocab_size: int,
        num_hidden: int,
        num_heads: int,
        num_hidden_feed_forward: int,
        num_layers: int,
        dropout: float,
        max_len: int = 1000,
    ) -> None:
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hidden)
        self.segment_embedding = nn.Embedding(2, num_hidden)
        self.layers = [
            nn.TransformerEncoderLayer(num_hidden, num_heads, num_hidden_feed_forward, dropout)
            for _ in range(num_layers)
        ]
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, num_hidden))

    def forward(
        self, tokens: List[str], segments: List[int], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Shape of X remains the same throughout this code block:
        # [batch_size, max_sequence_length, num_hiddens]
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X += self.positional_embedding[:, X.shape[1], :]
        for layer in self.layers:
            X = layer(X, mask)
        return X
