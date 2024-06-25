# import torch
import pytorch_lightning as pl

from torch import nn, optim
import torch.nn.functional as F


class NN(pl.LightningModule):
    def __init__(self, input_size, hidden_neurons, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, num_classes)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _common_block(self, batch, batch_inx):
        x, y = batch
        x = x.reshape(x.size(0), -1)  # Flatten the MNIST image
        scores = self.forward(x)
        loss = self.loss_function(scores, y)
        return loss, scores, y

    def training(self, batch, batch_inx):
        loss, scores, y = self._common_block(batch, batch_inx)
        return loss

    def validation(self, batch, batch_inx):
        loss, scores, y = self._common_block(batch, batch_inx)
        return loss

    def test(self, batch, batch_inx):
        loss, scores, y = self._common_block(batch, batch_inx)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
