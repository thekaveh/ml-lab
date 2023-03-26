import torch
from torch import nn

from consts import Consts

class NeuralNetModule(nn.Module):
    def __init__(self, dropout_p: float = Consts.DROPOUT_P):
        super(NeuralNetModule, self).__init__()

        self.dropout_p = dropout_p
        self.feature_size_in = Consts.FEATURES_SIZE_IN
        self.feature_size_out = Consts.FEATURES_SIZE_OUT

        self.L0 = nn.Dropout(p=self.dropout_p)

        self.L1 = nn.Linear(
            in_features=self.feature_size_in
            , out_features=self.feature_size_out
        )

        self.L2 = nn.ReLU()

        self.L3 = nn.LogSoftmax(dim=1)

    def forward(self, X):
        X = X.view(X.shape[0], -1)

        A0 = self.L0(X)
        A1 = self.L1(A0)
        A2 = self.L2(A1)
        A3 = self.L3(A2)

        return A3