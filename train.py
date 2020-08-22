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

import const
import model

BASE_LOG_PATH = path.join(const.BASE_DIR, 'log')


class BERTTrainer(object):
    def __init__(
        self,
        net: model.BERTModel,
        dataloader: utils_data.DataLoader,
        vocab: vocab.Vocab,
        optimizer: optim.Optimizer,
    ) -> None:
        self.net = net
        self.dataloader = dataloader
        self.vocab = vocab
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()

    def train_epochs(self, epochs: int, nsp_mult: float = 1.0):
        log_folder = self.setup_logging()

        epoch = 0
        batch_losses_mlm = []
        batch_losses_nsp = []
        while epoch <= epochs:
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
            pickle.dump(batch_losses_mlm, open(path.join(log_folder, 'batch_losses_mlm.p'), 'wb'))
            pickle.dump(batch_losses_nsp, open(path.join(log_folder, 'batch_losses_nsp.p'), 'wb'))
            print(f"Epoch took {time.time()-epoch_start_time:.1f} seconds")
            epoch += 1

    def setup_logging(self) -> str:
        now_str = str(datetime.now())
        log_folder = path.join(BASE_LOG_PATH, now_str)
        os.mkdir(log_folder)
        return log_folder

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
        net: model.BERTFineTuningModel,
        dataloader: utils_data.DataLoader,
        optimizer: optim.Optimizer,
    ) -> None:
        self.net = net
        self.dataloader = dataloader
        self.vocab = vocab
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()

    def train_epochs(self, epochs: int):
        log_folder = self.setup_logging()

        epoch = 0
        batch_losses = []
        while epoch <= epochs:
            epoch_start_time = time.time()
            for (examples_X, weights_X, segments_X, labels_y) in self.dataloader:
                self.optimizer.zero_grad()
                print("About to run batch...")
                batch_loss = self._get_batch_loss(examples_X, weights_X, segments_X, labels_y)
                print(f"Done! Loss {batch_loss.item():.4f}")
                batch_losses.append(batch_loss.item())
                batch_loss.backward()
                self.optimizer.step()
            pickle.dump(batch_losses, open(path.join(log_folder, 'batch_losses_mlm.p'), 'wb'))
            print(f"Epoch took {time.time()-epoch_start_time:.1f} seconds")
            epoch += 1

    def setup_logging(self) -> str:
        now_str = str(datetime.now())
        log_folder = path.join(BASE_LOG_PATH, 'fine_tuning', now_str)
        os.mkdir(log_folder)
        return log_folder

    def _get_batch_loss(
        self,
        tokens_X: torch.Tensor,
        weights_X: torch.Tensor,
        segments_X: torch.Tensor,
        labels_y: torch.Tensor,
    ) -> typing.Tuple:
        out = self.net(tokens_X, weights_X, segments_X)
        return self.loss(out, labels_y)
