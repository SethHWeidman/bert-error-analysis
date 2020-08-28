import torch
import transformers

from data import data
import model
import train


RANDOM_SEED = 200821


if __name__ == '__main__':
    batch_size, max_len = 64, 80

    print("Loading in pre-trained BERT model")
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = model.BERTFineTuningModel(RANDOM_SEED)
    sentiment_analysis_iter = data.load_sentiment_analysis_data(
        tokenizer, batch_size, max_len, None, False, 1
    )
    trainer = train.BERTFineTuningTrainerFromPretrained(
        bert_model,
        sentiment_analysis_iter,
        torch.optim.Adam(bert_model.parameters(), lr=0.0001),
        'fine_tune_pretrained',
        '02_five_epoch_fine_tuning_train_only',
    )
    print("Starting to train")
    trainer.train_epochs(5)
