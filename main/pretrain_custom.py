import time

import torch

from data import data
import model
import train


RANDOM_SEED = 200720


if __name__ == '__main__':
    print("Loading in Wiki2 data")
    print()
    start_time = time.time()
    batch_size, max_len = 512, 64
    train_iter, vocab, _ = data.load_wiki2_data(batch_size, max_len)
    print(f"Done! Took {time.time()-start_time:.0f} seconds")

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
        random_seed=RANDOM_SEED,
    )
    print("Starting to train BERTModel")
    print()
    trainer = train.BERTTrainer(
        net, train_iter, vocab, torch.optim.Adam(net.parameters(), lr=0.001)
    )
    trainer.train_epochs(50)