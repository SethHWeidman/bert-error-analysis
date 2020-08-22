'''
All models here taken from https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html
with minimal modification
'''

import typing

import torch
from torch import nn
from torch.nn import functional
import transformers


class BERTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_hidden: int,
        num_heads: int,
        num_hidden_feed_forward: int,
        num_layers: int,
        dropout: float,
        max_len: int,
        random_seed: int,
    ):
        super(BERTModel, self).__init__()
        torch.manual_seed(random_seed)
        self.encoder = BERTEncoder(
            vocab_size,
            num_hidden,
            num_heads,
            num_hidden_feed_forward,
            num_layers,
            dropout,
            max_len,
        )
        self.mlm = MaskLM(vocab_size, num_hidden)
        self.nsp = NextSentencePred(num_hidden)

    def forward(
        self,
        tokens: torch.Tensor,
        segments: torch.Tensor,
        valid_lens: typing.Optional[torch.tensor] = None,
        pred_positions: typing.Optional[torch.tensor] = None,
    ) -> typing.Tuple:
        encoded_X = self.encoder(tokens, segments, valid_lens)
        mlm_yhat = None if pred_positions is None else self.mlm(encoded_X, pred_positions)
        nsp_yhat = self.nsp(encoded_X[:, 0, :])
        return encoded_X, mlm_yhat, nsp_yhat


class BERTFineTuningModel(nn.Module):
    def __init__(self):
        super(BERTFineTuningModel, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 2)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        output = self.bert_model(input_ids, attention_mask, token_type_ids)[1]
        return self.fc(output)


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
        self.max_len = max_len
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

        X += self.positional_embedding[:, X.shape[1] - 1, :]

        if mask is not None:
            mask = _gen_uint8_tensor_mask_batch(mask, self.max_len)
        for layer in self.layers:
            # transpose since the PyTorch layer takes in X sequence_length first
            X_trans = X.transpose(1, 0)
            X_trans = layer(X_trans, src_key_padding_mask=mask)
        # transpose back
        return X_trans.transpose(1, 0)


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


class NextSentencePred(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super(NextSentencePred, self).__init__()
        self.hidden = nn.Linear(num_hidden, num_hidden)
        self.output = nn.Linear(num_hidden, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(X)))


def _gen_uint8_tensor_mask_batch(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.cat([_gen_bool_tensor_mask_obs(el.item(), max_len) for el in tensor])


def _gen_bool_tensor_mask_obs(el_length: int, max_len: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros([el_length], dtype=torch.bool),
            torch.ones([max_len - el_length], dtype=torch.bool),
        ]
    ).reshape(1, -1)
