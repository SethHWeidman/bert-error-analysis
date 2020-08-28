import time
import os
from os import path
import const

import torch
import pandas as pd
import transformers
from torch.nn import functional

from data import data
import model
import train


RANDOM_SEED = 200828

BASE_MODEL_PATH = path.join(const.BASE_DIR, 'models')

net = model.BERTFineTuningModel(RANDOM_SEED)

net.load_state_dict(
    torch.load(
        path.join(BASE_MODEL_PATH, 'fine_tune_pretrained/01_five_epoch_fine_tuning/model_epoch_1')
    )
)
net.eval()

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_analysis_iter = data.load_sentiment_analysis_data(tokenizer, 64, 80, None, False)

num_obs = 0
num_correct = 0
probs_positive = []
labels = []
for (examples_X, weights_X, segments_X, labels_y) in sentiment_analysis_iter:
    res = net(examples_X, weights_X, segments_X)
    pos_predictions = functional.softmax(res, dim=1)[:, 1]
    probs_positive.append(pos_predictions.data)
    labels.append(labels_y)
    acc_tensor = torch.max(functional.softmax(res, dim=1), dim=1).indices == labels_y
    num_correct_batch = acc_tensor.type(torch.ByteTensor).sum().item()
    num_in_batch = acc_tensor.shape[0]
    num_correct += num_correct_batch
    num_obs += num_in_batch
    print(f"Accuracy for batch: {num_correct_batch}/{num_in_batch}")

pd.DataFrame({'probs': torch.cat(probs_positive), 'labels': torch.cat(labels)}).to_csv(
    'eval/whole_dataset_in_sample_accuracy.csv', index=False
)
