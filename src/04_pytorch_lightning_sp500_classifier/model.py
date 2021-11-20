# =====================================================================================================================
# Aufgabe S&P 500 Classifier with PyTorch Lightning
# 20.11.2021, Thomas Iten
# =====================================================================================================================

import torch, torchmetrics
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F

class SP500ClassifierModel(pl.LightningModule):

    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.metric = torchmetrics.Accuracy()
        #
        # Model experiments
        #
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(489, 512),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Model experiment 3
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(489, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        # Model experiment 2
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(489 , 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1))
        # Model experiment 1
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(489, 512),
        #     nn.Dropout(.5),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.Dropout(.5),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.Dropout(.5),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        #
        # Loss experiments
        #
        self.loss_fn = F.binary_cross_entropy_with_logits
        # Loss experiemnt 1
        # self.loss_fn = nn.BCEWithLogitsLoss()

    def process_step(self, batch, batch_idx):
        x, y = batch
        pred = self.linear_relu_stack(x)
        loss = self.loss_fn(pred, y)

        # calculate accuracy
        pred = (pred>0.5).int()
        acc = self.metric(pred, y.int())

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.process_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.process_step(batch, batch_idx)
        print("val_loss:", loss)
        self.log('val_accuracy', acc)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss, acc = self.process_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

    def configure_optimizers(self):
        # Update params step: params = params - lr * grad
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
