'''
All models here taken from https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html 
with minimal modification
'''

import typing

import torch
from torch import nn
from torch.nn import functional


class BERTEncoder(nn.Module):
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
        self,
        tokens: typing.List[str],
        segments: typing.List[int],
        mask: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Shape of X remains the same throughout this code block:
        # [batch_size, max_sequence_length, num_hiddens]
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X += self.positional_embedding[:, X.shape[1], :]
        for layer in self.layers:
            X = layer(X, mask)
        return X


class MaskLM(nn.Module):
    def __init__(self, vocab_size: int, num_hidden: int) -> None:
        super(MaskLM, self).__init__()
        self.dense1 = nn.Linear(num_hidden, num_hidden)
        self.layer_norm = nn.LayerNorm(num_hidden)
        self.dense2 = nn.Linear(num_hidden, vocab_size)

    def forward(self, X: torch.Tensor, pred_positions: torch.Tensor) -> torch.Tensor:
        num_pred_positions = pred_positions.shape[1]

        # done to make the selection of elements from X below easier
        pred_positions = pred_positions.reshape(-1)

        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)

        # selection of the elements to be predicted on from X
        # first row: batch element 0, first element in pred_positions
        # second row: batch element 0, second element in pred_positions
        # ...
        # last row: last batch element, last element in pred_positions
        # each row will have dimension hidden_size
        masked_X = X[torch.repeat_interleave(batch_idx, num_pred_positions), pred_positions]

        # reshape to be batch_size, num_pred_positions, num_hidden
        masked_X = masked_X.reshape(batch_size, num_pred_positions, -1)

        return self.dense2(self.layer_norm(functional.relu(self.dense1(masked_X))))
