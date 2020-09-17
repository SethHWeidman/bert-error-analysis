from datetime import datetime
import pickle
import os
from os import path
import time
import typing

import torch
from torch import nn
from torch.utils import data as utils_data
from torch import optim
from torchtext import vocab
from transformers import BertForSequenceClassification

import const
import model

BASE_LOG_PATH = path.join(const.BASE_DIR, 'log')
BASE_MODEL_PATH = path.join(const.BASE_DIR, 'models')


class BERTTrainer(object):
    def __init__(
        self,
        net: model.BERTModel,
        dataloader: utils_data.DataLoader,
        vocab: vocab.Vocab,
        optimizer: optim.Optimizer,
        training_type: typing.Optional[str] = '',
        run_name: typing.Optional[str] = '',
    ) -> None:
        self.net = net
        self.dataloader = dataloader
        self.vocab = vocab
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()
        self.training_type = training_type
        self.run_name = run_name
        self.log_folder = self.setup_folder(BASE_LOG_PATH)
        self.model_folder = self.setup_folder(BASE_MODEL_PATH)

    def train_epochs(self, epochs: int, nsp_mult: float = 1.0):

        epoch = 0
        batch_losses_mlm = []
        batch_losses_nsp = []
        while epoch < epochs:
            epoch_start_time = time.time()
            for (
                tokens_X,
                segments_X,
                valid_lens_X,
                pred_positions_X,
                mlm_weights_X,
                mlm_y,
                nsp_y,
            ) in self.dataloader:
                self.optimizer.zero_grad()
                mlm_loss, nsp_loss = self._get_batch_loss_bert(
                    tokens_X,
                    segments_X,
                    valid_lens_X,
                    pred_positions_X,
                    mlm_weights_X,
                    mlm_y,
                    nsp_y,
                )
                batch_losses_mlm.append(mlm_loss.item())
                batch_losses_nsp.append(nsp_loss.item())
                total_loss = mlm_loss + nsp_loss * nsp_mult
                total_loss.backward()
                self.optimizer.step()
            pickle.dump(
                batch_losses_mlm, open(path.join(self.log_folder, 'batch_losses_mlm.p'), 'wb')
            )
            pickle.dump(
                batch_losses_nsp, open(path.join(self.log_folder, 'batch_losses_nsp.p'), 'wb')
            )
            torch.save(
                self.net.state_dict(), path.join(self.model_folder, f'model_epoch_{epoch+1}')
            )
            print(f"Epoch took {time.time()-epoch_start_time:.1f} seconds")
            epoch += 1

    def setup_folder(self, base_path: str) -> str:
        run_str = self.run_name if len(self.run_name) else str(datetime.now())
        parent_folder = path.join(base_path, self.training_type)
        if not path.isdir(parent_folder):
            os.mkdir(parent_folder)
        folder = path.join(parent_folder, run_str)
        os.mkdir(folder)
        return folder

    def load_model(self, epoch_num: int) -> model.BERTModel:
        new_model = model.BERTModel(
            len(self.vocab),
            num_hidden=128,
            num_heads=2,
            num_hidden_feed_forward=256,
            num_layers=2,
            dropout=0.2,
            max_len=64,
        )
        new_model.load_state_dict(
            torch.load(path.join(self.model_folder, f'model_epoch_{epoch_num}.pt'))
        )
        return new_model

    def _get_batch_loss_bert(
        self,
        tokens_X: torch.Tensor,
        segments_X: torch.Tensor,
        valid_lens_X: torch.Tensor,
        pred_positions_X: torch.Tensor,
        mlm_weights_X: torch.Tensor,
        mlm_y: torch.Tensor,
        nsp_y: torch.Tensor,
    ) -> typing.Tuple:

        _, mlm_yhat, nsp_yhat = self.net(
            tokens_X, segments_X, valid_lens_X.reshape(-1), pred_positions_X
        )

        # reshape and multiply to zero out the masked positions
        mlm_yhat_reshaped = mlm_yhat.reshape((-1, len(self.vocab)))
        pred_positions_reshaped = mlm_weights_X.reshape(-1, 1)
        mlm_yhat_preds = mlm_yhat_reshaped * pred_positions_reshaped

        mlm_loss = self.loss(mlm_yhat_preds, mlm_y.reshape(-1))
        mlm_loss_final = mlm_loss.sum()

        nsp_loss = self.loss(nsp_yhat, nsp_y)
        nsp_loss_final = nsp_loss.mean()
        return mlm_loss_final, nsp_loss_final


class BERTFineTuningTrainerFromPretrained(object):
    def __init__(
        self,
        net: typing.Union[model.BERTFineTuningModel, BertForSequenceClassification],
        dataloader: utils_data.DataLoader,
        optimizer: optim.Optimizer,
        training_type: typing.Optional[str] = '',
        run_name: typing.Optional[str] = '',
    ) -> None:
        self.net = net
        self.dataloader = dataloader
        self.vocab = vocab
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()
        self.training_type = training_type
        self.run_name = run_name
        self.log_folder = self.setup_folder(BASE_LOG_PATH)
        self.model_folder = self.setup_folder(BASE_MODEL_PATH)

    def train_epochs(self, epochs: int):
        epoch = 0
        batch_losses = []
        while epoch < epochs:
            epoch_start_time = time.time()
            for (examples_X, weights_X, segments_X, labels_y) in self.dataloader:
                self.optimizer.zero_grad()
                batch_loss = self._get_batch_loss(examples_X, weights_X, segments_X, labels_y)
                batch_losses.append(batch_loss.item())
                batch_loss.backward()
                self.optimizer.step()
            pickle.dump(batch_losses, open(path.join(self.log_folder, 'batch_losses.p'), 'wb'))
            torch.save(
                self.net.state_dict(), path.join(self.model_folder, f'model_epoch_{epoch+1}')
            )
            print(f"Epoch took {time.time()-epoch_start_time:.1f} seconds")
            epoch += 1

    def setup_folder(self, base_path: str) -> str:
        run_str = self.run_name if len(self.run_name) else str(datetime.now())
        parent_folder = path.join(base_path, self.training_type)
        if not path.isdir(parent_folder):
            os.mkdir(parent_folder)
        folder = path.join(parent_folder, run_str)
        if not path.isdir(folder):
            os.mkdir(folder)
        return folder

    def _get_batch_loss(
        self,
        tokens_X: torch.Tensor,
        weights_X: torch.Tensor,
        segments_X: torch.Tensor,
        labels_y: torch.Tensor,
    ) -> typing.Tuple:
        out = self.net(tokens_X, weights_X, segments_X)
        if isinstance(self.net, BertForSequenceClassification):
            return self.loss(out[0], labels_y)
        else:
            return self.loss(out, labels_y)
        


def get_most_recent_dir(folder: str) -> str:
    return max(os.listdir(folder))
