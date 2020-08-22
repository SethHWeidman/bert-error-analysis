import torch
import transformers

from data import data
import model
import train


RANDOM_SEED = 200720


if __name__ == '__main__':
    batch_size, max_len = 512, 64

    print("Loading in pre-trained BERT model")
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = model.BERTFineTuningModel()
    sentiment_analysis_iter = data.load_sentiment_analysis_data(
        tokenizer, batch_size, max_len, None, False
    )
    trainer = train.BERTFineTuningTrainerFromPretrained(
        bert_model, sentiment_analysis_iter, torch.optim.Adam(bert_model.parameters(), lr=0.001)
    )
    print("Starting to train")
    trainer.train_epochs(50)