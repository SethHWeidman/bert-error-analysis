import torch

import model
import preprocess

RANDOM_SEED = 200720

if __name__ == '__main__':
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