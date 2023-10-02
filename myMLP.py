from torch import nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics import Accuracy

number_of_features = 22

# softmax for multiclass

class MLP(pl.LightningModule):

    def __init__(self):
        super().__init__()
        input_size = number_of_features
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.ce = nn.BCELoss()
        self.train_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        labels = y.unsqueeze(1)
        output = self.layers(x)
        loss = self.ce(output, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.train_acc.update(output, labels)
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        labels = y.unsqueeze(1)
        output = self.layers(x)
        loss = self.ce(output, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.test_acc.update(output, labels)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3) # lr=1e-4
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5)
        return optimizer
        # return [optimizer], [scheduler]

# https://odsc.medium.com/higher-level-pytorch-apis-a-short-introduction-to-pytorch-lightning-47b31c7a45de
# https://lightning.ai/courses/deep-learning-fundamentals/training-multilayer-neural-networks-overview/4-3-training-a-multilayer-neural-network-in-pytorch-part-1-5/
