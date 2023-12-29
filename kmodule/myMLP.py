from torch import nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

# number_of_features = 22

# softmax for multiclass

class MLP(pl.LightningModule):

    def __init__(self, number_of_features):
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
        self.loss_per_step = []
        self.loss_per_epo = []
        self.acc_per_step = []
        self.acc_per_epo = []

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        labels = y.unsqueeze(1)
        output = self.layers(x)
        loss = self.ce(output, labels)
        self.loss_per_step.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        self.train_acc.update(output, labels)
        val = self.train_acc.compute()
        self.acc_per_step.append(val)
        self.log("train_acc", val, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.loss_per_epo.append(sum(self.loss_per_step) / len(self.loss_per_step))
        self.loss_per_step.clear()
        self.acc_per_epo.append((sum(self.acc_per_step) / len(self.acc_per_step)).cpu())
        self.acc_per_step.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        labels = y.unsqueeze(1)
        output = self.layers(x)
        loss = self.ce(output, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.test_acc.update(output, labels)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return output, loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3) # lr=1e-4
        # scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, step_size=10, gamma=0.1)
        return optimizer
        # return [optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx: int = None , dataloader_idx: int = None):
        x, _ = batch
        return self(x)
    
    def plotting(self):
        plt.figure(figsize=(10,5))
        plt.title("Training Loss")
        plt.plot(self.loss_per_epo)
        plt.xlabel("epoches")
        plt.ylabel("loss")
        plt.savefig('loss_func.png')
        plt.show()

        plt.figure(figsize=(10,5))
        plt.title("Training Accuracy")
        plt.plot(self.acc_per_epo)
        plt.xlabel("epoches")
        plt.ylabel("accuracy")
        plt.savefig('acc_func.png')
        plt.show()

# https://odsc.medium.com/higher-level-pytorch-apis-a-short-introduction-to-pytorch-lightning-47b31c7a45de
# https://lightning.ai/courses/deep-learning-fundamentals/training-multilayer-neural-networks-overview/4-3-training-a-multilayer-neural-network-in-pytorch-part-1-5/
