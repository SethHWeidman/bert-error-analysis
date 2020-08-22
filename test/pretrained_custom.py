import time

import torch

import model


RANDOM_SEED = 200720


if __name__ == '__main__':
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