import os
from os import path
import typing

from allennlp.predictors.predictor import Predictor
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.utils import data as utils_data
from transformers import BertTokenizer, RobertaTokenizer

import const
from data import data, sentiment_analysis
import model


RANDOM_SEED = 200828

BASE_MODEL_PATH = path.join(const.BASE_DIR, 'models')


def compute_accuracy_df(
    model_type: str,
    training_run: str,
    epoch: int,
    use_binary_labels: bool = True,
    add_lstm_preds: bool = False,
    use_roberta: bool = False,
) -> None:
    eval_dir_part = path.join('eval', model_type)
    if not path.exists(eval_dir_part):
        os.mkdir(eval_dir_part)
    eval_dir_whole = path.join(eval_dir_part, training_run)
    if not path.exists(eval_dir_whole):
        os.mkdir(eval_dir_whole)

    if not use_roberta:
        net = model.BERTFineTuningModel(2 if use_binary_labels else 3, RANDOM_SEED)
    else:
        net = model.RobertaFineTuningModel(2 if use_binary_labels else 3, RANDOM_SEED)
    net.load_state_dict(
        torch.load(path.join(BASE_MODEL_PATH, model_type, training_run, f'model_epoch_{epoch}'))
    )
    net.eval()
    tokenizer = (
        BertTokenizer.from_pretrained('bert-base-uncased')
        if not use_roberta
        else RobertaTokenizer.from_pretrained('roberta-base')
    )

    max_len = 80 if not use_roberta else 63
    sentiment_analysis_dataloader = data.load_sentiment_analysis_data(
        tokenizer, 64, max_len, None, 2, use_binary_labels, False,
    )

    if use_binary_labels:
        probs_positive, labels = compute_accuracy_two_class(sentiment_analysis_dataloader, net)
    else:
        probs_positive, labels = compute_accuracy_three_class(
            sentiment_analysis_dataloader, net, use_roberta
        )

    probs_str = 'bert_probs' if not use_roberta else 'roberta_probs'
    df = pd.DataFrame({probs_str if add_lstm_preds else 'probs': probs_positive, 'labels': labels})

    dataset = sentiment_analysis.SentimentAnalysisDataset(tokenizer, max_len, split_to_use=2)
    assert len(dataset.sentences) == df.shape[0]
    df['sentence'] = dataset.sentences

    if add_lstm_preds:
        predictor_lstm = Predictor.from_path(const.LSTM_PATH)
        preds_lstm = [
            predictor_lstm.predict(sentence=dataset_sentence)['probs'][0]
            for dataset_sentence in dataset.sentences
        ]
        df['lstm_probs'] = preds_lstm

    df.to_csv(path.join(eval_dir_whole, f'test_accuracy_epoch_{epoch}.csv'), index=False)


def compute_accuracy_two_class(dataloader: utils_data.DataLoader, net: nn.Module) -> typing.Tuple:
    probs_positive = []
    labels = []
    for (examples_X, weights_X, segments_X, labels_y) in dataloader:
        res = net(examples_X, weights_X, segments_X)
        pos_predictions = functional.softmax(res, dim=1)[:, 1]
        probs_positive.append(pos_predictions.data)
        labels.append(labels_y)
        acc_tensor = torch.max(functional.softmax(res, dim=1), dim=1).indices == labels_y
        num_correct_batch = acc_tensor.type(torch.ByteTensor).sum().item()
        num_in_batch = acc_tensor.shape[0]
        print(f"Accuracy for batch: {num_correct_batch}/{num_in_batch}")
    return torch.cat(probs_positive), torch.cat(labels)


def compute_accuracy_three_class(
    dataloader: utils_data.DataLoader, net: nn.Module, use_roberta: bool = False
) -> typing.Tuple:
    probs_positive = []
    labels = []
    for batch in dataloader:
        if not use_roberta:
            examples_X, weights_X, segments_X, labels_y = batch
            res = net(examples_X, weights_X, segments_X)
        else:
            examples_X, weights_X, labels_y = batch
            res = net(examples_X, weights_X)
        pos_predictions = model_res_to_final_pos_preds(res)
        probs_positive.append(pos_predictions)
        labels.append(labels_y)
        acc_tensor = final_preds_to_labels(pos_predictions, labels_y)
        num_correct_batch = acc_tensor.type(torch.ByteTensor).sum().item()
        num_in_batch = acc_tensor.shape[0]
        print(f"Accuracy for batch: {num_correct_batch}/{num_in_batch}")
    return torch.cat(probs_positive), torch.cat(labels)


def model_res_to_final_pos_preds(res: torch.tensor) -> torch.tensor:
    preds = []
    for row in res:
        preds.append(functional.softmax(row[[0, 2]], dim=0))
    return torch.stack(preds).data[:, 1]


def final_preds_to_labels(
    pos_preds_tensor: torch.tensor, labels_tensor: torch.tensor
) -> torch.Tensor:
    non_neutral_labels_mask = labels_tensor != 1
    pos_preds_tensor_subset = pos_preds_tensor[non_neutral_labels_mask]
    labels_tensor_subset = labels_tensor[non_neutral_labels_mask]
    return (pos_preds_tensor_subset > 0.5) == (labels_tensor_subset == 2)


def return_accuracy_csv(filepath: str) -> float:
    df = pd.read_csv(filepath)
    df = df[df['labels'] != 1]
    df['preds'] = df['probs'].apply(lambda x: 2 if x > 0.5 else 0)
    return (df['preds'] == df['labels']).mean()


if __name__ == '__main__':
    # compute_accuracy_df(
    #     'roberta_fine_tune_pretrained_three_class',
    #     '01_five_epoch_fine_tuning_train_only_lr2e-5_bs16',
    #     1,
    #     False,
    #     True,
    #     True,
    # )

    # compute_accuracy_df(
    #     'roberta_fine_tune_pretrained_three_class',
    #     '01_five_epoch_fine_tuning_train_only_lr2e-5_bs16',
    #     2,
    #     False,
    #     True,
    #     True,
    # )

    compute_accuracy_df(
        'roberta_fine_tune_pretrained_three_class',
        '01_five_epoch_fine_tuning_train_only_lr2e-5_bs16',
        4,
        False,
        True,
        True,
    )