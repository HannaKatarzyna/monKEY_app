import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torchvision import transforms
from torchmetrics import Accuracy

class autoEncoder(nn.Module):
    def __init__(self, n_embedded, X_train):
        super(autoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X_train.shape[1], n_embedded),
            nn.ReLU()                 
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_embedded, X_train.shape[1])
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_embedded, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Dropout(0.2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        out = self.classifier(encoded)
        return decoded, out
    
model = autoEncoder(40)
            
criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()
            
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 150
   
for epoch in range(epochs): 
    for inputs, labels in train_loader:
      
        optimizer.zero_grad()
        decoded, out = model(inputs)
       
        loss1 = criterion1(decoded, inputs) 
        loss2 = criterion2(out, labels)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()