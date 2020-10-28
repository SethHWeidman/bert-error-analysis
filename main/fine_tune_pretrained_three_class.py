import torch
from transformers import BertTokenizer, RobertaTokenizer

from data import data
import model
import train


if __name__ == '__main__':
    batch_size = 16

    # print("Loading in pre-trained BERT model")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = model.RobertaFineTuningModel(3)
    # sentiment_analysis_dataloader = data.load_sentiment_analysis_data(
    #     tokenizer, batch_size, max_len, None, False, 1, False
    # )
    # trainer = train.BERTFineTuningTrainerFromPretrained(
    #     bert_model,
    #     sentiment_analysis_dataloader,
    #     torch.optim.Adam(bert_model.parameters(), lr=5e-5),
    #     'fine_tune_pretrained_three_class',
    #     '02_five_epoch_fine_tuning_train_only_lr5e-5',
    # )
    # print("Starting to train")
    # trainer.train_epochs(1)

    print("Loading in pre-trained BERT model")
    max_len = 63
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = model.RobertaFineTuningModel(3)
    sentiment_analysis_dataloader = data.load_sentiment_analysis_data(
        tokenizer, batch_size, max_len, None, 1, False
    )
    trainer = train.BERTFineTuningTrainerFromPretrained(
        roberta_model,
        sentiment_analysis_dataloader,
        torch.optim.Adam(roberta_model.parameters(), lr=2e-5),
        'roberta_fine_tune_pretrained_three_class',
        '01_five_epoch_fine_tuning_train_only_lr2e-5_bs16',
    )
    print("Starting to train")
    trainer.train_epochs(5)
