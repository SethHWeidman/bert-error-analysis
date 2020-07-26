import typing

import torch
from torch import nn
from torch.utils import data as utils_data
from torch import optim
from torchtext import vocab

import model


class BERTTrainer(object):
    def __init__(
        self,
        net: model.BERTModel,
        dataloader: utils_data.DataLoader,
        vocab: vocab.Vocab,
        optimizer: optim.Optimizer,
        batch_size: int,
        max_len: int,
    ) -> None:
        self.net = net
        self.dataloader = dataloader
        self.vocab = vocab
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()

    def train_epochs(self, epochs: int):
        epoch = 0
        while epoch <= epochs:
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
                print("MLM loss for batch", mlm_loss.item())
                print("MLM loss for batch", nsp_loss.item())
                total_loss = mlm_loss + nsp_loss
                total_loss.backward()
                self.optimizer.step()

            epochs += 1

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
        mlm_loss_final = mlm_loss.sum() / (mlm_weights_X.sum() + 1e-8)

        nsp_loss = self.loss(nsp_yhat, nsp_y)
        nsp_loss_final = nsp_loss.mean()
        return mlm_loss_final, nsp_loss_final
