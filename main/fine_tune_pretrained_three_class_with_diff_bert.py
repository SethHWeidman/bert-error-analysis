import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from data import data
import model
import train

if __name__ == '__main__':
    batch_size, max_len = 32, 80

    print("Loading in pre-trained BERT model")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig(model_type='bert-base-uncased', num_labels=3)
    bert_model = BertForSequenceClassification(config)
    sentiment_analysis_dataloader = data.load_sentiment_analysis_data(
        tokenizer, batch_size, max_len, None, False, 1, False
    )
    trainer = train.BERTFineTuningTrainerFromPretrained(
        bert_model,
        sentiment_analysis_dataloader,
        torch.optim.Adam(bert_model.parameters(), lr=5e-5),
        'fine_tune_pretrained_three_class',
        '03_five_epoch_fine_tuning_diff_bert_lr5e-5',
    )
    print("Starting to train")
    trainer.train_epochs(3)

    trainer = train.BERTFineTuningTrainerFromPretrained(
        bert_model,
        sentiment_analysis_dataloader,
        torch.optim.Adam(bert_model.parameters(), lr=4e-5),
        'fine_tune_pretrained_three_class',
        '03_five_epoch_fine_tuning_diff_bert_lr4e-5',
    )
    print("Starting to train")
    trainer.train_epochs(3)

    trainer = train.BERTFineTuningTrainerFromPretrained(
        bert_model,
        sentiment_analysis_dataloader,
        torch.optim.Adam(bert_model.parameters(), lr=3e-5),
        'fine_tune_pretrained_three_class',
        '03_five_epoch_fine_tuning_diff_bert_lr3e-5',
    )
    print("Starting to train")
    trainer.train_epochs(3)

    trainer = train.BERTFineTuningTrainerFromPretrained(
        bert_model,
        sentiment_analysis_dataloader,
        torch.optim.Adam(bert_model.parameters(), lr=2e-5),
        'fine_tune_pretrained_three_class',
        '03_five_epoch_fine_tuning_diff_bert_lr2e-5',
    )
    print("Starting to train")
    trainer.train_epochs(3)            
