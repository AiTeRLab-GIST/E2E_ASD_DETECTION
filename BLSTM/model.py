import pdb
import conf
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple

class lstm_block(nn.Module):
    def __init__(self, in_shape, hidden_size, batch_first):
        super().__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(in_shape, hidden_size, num_layers = 1, batch_first = True, bidirectional = True)
        self.BN = nn.BatchNorm1d(2 * hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        if len(x.shape)>2:
            x = self.BN(x.permute(0,2,1)).permute(0,2,1)
        else:
            x = self.BN(x)
        return x

class dense_block(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.linear = nn.Linear(in_shape, out_shape)
        self.BN = nn.BatchNorm1d(out_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if len(x.shape) == 3:
            x = self.BN(x.permute(0,2,1)).permute(0,2,1)
        else:
            x = self.BN(x)
        x = self.relu(x)
        return x

class BLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.aux = nn.Linear(88, 128)
        self.lstm = lstm_block(128, 128, True)
        self.dense1a = dense_block(256, 128)
        self.dense2a = dense_block(128, 64)
        self.dense3a = dense_block(64, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.aux(x)
        x = self.lstm(x)
        x = self.dense1a(x)
        x = self.dense2a(x)
        x = self.dense3a(x)
        x = nn.MaxPool1d(x.shape[1])(x.transpose(2,1)).squeeze()
        return self.softmax(x) 