import time

import torch

from data import data


if __name__ == '__main__':
    batch_size, max_len = 512, 64
    _, vocab, tokenizer = data.load_wiki2_data(batch_size, max_len)
    start_time = time.time()
    print("Testing Sentiment Analysis dataset")
    sentiment_analysis_iter = data.load_sentiment_analysis_data(
        tokenizer, batch_size, max_len, vocab
    )
    for (examples_X, weights_X, segments_X, labels_y) in sentiment_analysis_iter:
        assert examples_X.shape == torch.Size([512, 64])
        assert weights_X.shape == torch.Size([512, 64])
        assert segments_X.shape == torch.Size([512, 64])
        assert labels_y.shape == torch.Size([512])
        break
    print(f"All done! Took {time.time()-start_time:.0f} seconds")