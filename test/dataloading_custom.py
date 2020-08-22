import time

import torch

from data import data


RANDOM_SEED = 200720


if __name__ == '__main__':
    start_time = time.time()
    print("Testing data loading")
    batch_size, max_len = 512, 64
    train_iter, vocab, tokenizer = data.load_wiki2_data(batch_size, max_len)
    for (
        tokens_X,
        segments_X,
        valid_lens_X,
        pred_positions_X,
        mlm_weights_X,
        mlm_y,
        nsp_y,
    ) in train_iter:
        assert tokens_X.shape == torch.Size([512, 64])
        assert segments_X.shape == torch.Size([512, 64])
        assert valid_lens_X.shape == torch.Size([512])
        assert pred_positions_X.shape == torch.Size([512, 10])
        assert mlm_weights_X.shape == torch.Size([512, 10])
        assert mlm_y.shape == torch.Size([512, 10])
        assert nsp_y.shape == torch.Size([512])
        assert len(vocab) == 17962
        break
    print(f"All done! Took {time.time()-start_time:.0f} seconds")