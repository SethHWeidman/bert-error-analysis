import os
import time
import typing

import torch
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
    print("Testing model")
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
