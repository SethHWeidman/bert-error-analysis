import time

import torch
import transformers

from data import data


if __name__ == '__main__':
    batch_size, max_len = 512, 80
    start_time = time.time()
    print("Testing Sentiment Analysis dataset with pre-trained tokenizer")
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    sentiment_analysis_iter = data.load_sentiment_analysis_data(
        tokenizer, batch_size, max_len, None, False
    )
    for examples_X, weights_X, segments_X, labels_y in sentiment_analysis_iter:
        print(examples_X.shape)
        assert examples_X.shape == torch.Size([batch_size, max_len])
        assert weights_X.shape == torch.Size([batch_size, max_len])
        assert segments_X.shape == torch.Size([batch_size, max_len])
        assert labels_y.shape == torch.Size([batch_size])
        break

    print(f"All done! Took {time.time()-start_time:.0f} seconds")
