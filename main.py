import os
import time
import typing

import torch
from torch import nn
from torch.utils import data as utils_data

import model
import preprocess
from data import data

RANDOM_SEED = 200720


def load_wiki2_data(batch_size: int, max_len: int) -> typing.Tuple:
    dataset = data.BERTWiki2Dataset(
        os.path.join(os.getcwd(), 'data', 'wikitext-2'), 50000, max_len
    )
    # https://pytorch.org/docs/stable/data.html
    train_iter = utils_data.DataLoader(dataset, batch_size=batch_size)
    return train_iter, dataset.vocab


if __name__ == '__main__':
    print("Starting to run tests")
    print()
    print("Testing BERTEncoder model")
    start_time = time.time()
    vocab_size, num_hidden, num_heads, num_hidden_feed_forward = 10000, 768, 2, 1024
    num_layers, dropout = 2, 0.2
    bert_encoder = model.BERTEncoder(
        vocab_size, num_hidden, num_heads, num_hidden_feed_forward, num_layers, dropout
    )

    # two "sentences", each with length 8
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])

    encoded = bert_encoder(tokens, segments, None)
    assert encoded.shape == torch.Size([2, 8, 768])
    print(f"All done! Took {time.time()-start_time:.0f} seconds")

    print("Testing MaskLM model")
    start_time = time.time()
    mlm = model.MaskLM(vocab_size, num_hidden)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_yhat = mlm(encoded, mlm_positions)
    assert mlm_yhat.shape == torch.Size([2, 3, 10000])
    print(f"All done! Took {time.time()-start_time:.0f} seconds")

    print("Testing NextSentencePred model")
    start_time = time.time()
    nsp = model.NextSentencePred(num_hidden)
    # '<cls>' token only
    nsp_yhat = nsp(encoded[:, 0, :])
    assert nsp_yhat.shape == torch.Size([2, 2])
    print(f"All done! Took {time.time()-start_time:.0f} seconds")

    print("Testing CrossEntropyLoss model")
    start_time = time.time()
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    mlm_loss = loss(mlm_yhat.reshape(-1, vocab_size), mlm_y.reshape(-1))
    assert mlm_loss.shape == torch.Size([6])
    print(f"All done! Took {time.time()-start_time:.0f} seconds")

    start_time = time.time()
    print("Testing data loading")
    batch_size, max_len = 512, 64
    train_iter, vocab = load_wiki2_data(batch_size, max_len)
    for (
        tokens_X,
        segments_X,
        valid_lens_X,
        pred_positions_X,
        mlm_weights_X,
        mlm_y,
        nsp_y,
    ) in train_iter:
        assert (
            tokens_X.shape,
            segments_X.shape,
            valid_lens_X.shape,
            pred_positions_X.shape,
            mlm_weights_X.shape,
            mlm_y.shape,
            nsp_y.shape,
        ) == (
            torch.Size([512, 64]),
            torch.Size([512, 64]),
            torch.Size([512]),
            torch.Size([512, 10]),
            torch.Size([512, 10]),
            torch.Size([512, 10]),
            torch.Size([512]),
        )
        assert len(vocab) == 18939
        break
    print(f"All done! Took {time.time()-start_time:.0f} seconds")

    # Actual BERT paper parameters:
    # * num_hidden = 768
    # * num_heads = 12
    # * num_hidden_feed_forward = 768
    # * num_layers = 12
    net = model.BERTModel(
        len(vocab),
        num_hidden=128,
        num_heads=2,
        num_hidden_feed_forward=256,
        num_layers=2,
        dropout=0.2,
        max_len=max_len,
    )
    # TODO: train model
