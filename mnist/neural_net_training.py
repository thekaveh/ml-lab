import torch
import numpy as np

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from consts import Consts
from neural_net_module import NeuralNetModule
from iteration_data_point import IterationDataPoint

class NeuralNetTraining():
    def __init__(
        self
        , loader_val: DataLoader
        , loader_train: DataLoader
        , etha: float = Consts.ETHA
        , epochs: int = Consts.NUM_EPOCHS
        , device_name: str = Consts.DEVICE
        , dropout_p: float = Consts.DROPOUT_P
        , optimizer_alg: str = Consts.OPTIMIZER_ALG
    ):
        self.loader_val = loader_val
        self.loader_train = loader_train

        self.etha = etha
        self.epochs = epochs
        self.dropout_p = dropout_p
        self.optimizer_alg = optimizer_alg

        self.device_name = device_name
        self.device = torch.device(device_name)

        self.module = NeuralNetModule(dropout_p=self.dropout_p).to(self.device)
        self.smce = nn.CrossEntropyLoss().to(self.device)

        if optimizer_alg == "sgd":
            self.optimizer = optim.SGD(
                lr=self.etha
                , params=self.module.parameters()
            )
        elif optimizer_alg == "adam":
            self.optimizer = optim.Adam(
                lr=self.etha
                , params=self.module.parameters()
            )

    def train_and_validate(self):
        iter_idx: int = 0
        iteration_data = []

        tqdm_bar = tqdm(total=int(self.epochs * len(self.loader_train.batch_sampler)))

        for epoch_idx in range(self.epochs):
            for mb_idx, (X, Y) in enumerate(self.loader_train):
                self.module.zero_grad()

                X = X.to(self.device)
                Y = Y.to(self.device)

                Y_hat = self.module(X)
                L_train = self.smce(Y_hat, Y)

                total_preds = 0
                correct_preds = 0
                for Y_hat_idx, y_hat in enumerate(Y_hat):
                    total_preds += 1
                    if torch.argmax(y_hat) == Y[Y_hat_idx]:
                        correct_preds += 1
                E_train = 1 - correct_preds / total_preds

                L_train.backward()
                self.optimizer.step()

                L_val, E_val = self._validate()

                idp = IterationDataPoint(
                    epoch_idx=epoch_idx
                    , mini_batch_idx=mb_idx
                    , iter_idx=iter_idx
                    , training_loss=float(L_train)
                    , training_error=E_train
                    , validation_loss=float(L_val)
                    , validation_error=E_val
                )

                iteration_data.append(idp)

                iter_idx += 1
                tqdm_bar.update(1)
        
        return np.array(iteration_data)

    def _validate(self):
        loss_vals, err_vals = [], []

        with torch.no_grad():
            for mb_idx, (X, Y) in enumerate(self.loader_val):
                X = X.to(self.device)
                Y = Y.to(self.device)

                Y_hat = self.module(X)
                L_val = self.smce(Y_hat, Y)
                
                loss_vals.append(L_val)

                total_preds = 0
                correct_preds = 0
                for Y_hat_idx, y_hat in enumerate(Y_hat):
                    total_preds += 1
                    if torch.argmax(y_hat) == Y[Y_hat_idx]:
                        correct_preds += 1
                E_val = 1 - correct_preds / total_preds

                err_vals.append(E_val)

        return (
            torch.mean(torch.FloatTensor(loss_vals)).item()
            , torch.mean(torch.FloatTensor(err_vals)).item()
        )
    
    def __str__(self):
        return (
            "device={device}, epoch(s)={epochs}, etha={etha}, optimizer={optimizer}, and dropout_p={dropout_p}"
            .format(
                etha=self.etha
                , epochs=self.epochs
                , device=self.device_name
                , dropout_p=self.dropout_p
                , optimizer=self.optimizer_alg
            )
        )