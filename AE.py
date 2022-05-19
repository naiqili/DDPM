import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import os

# Defining Model

class AE(nn.Module):
    def __init__(self, num_features, dimin=28 * 28, p=0.2):
        super(AE, self).__init__()
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(dimin, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2000)
        self.fc4 = nn.Linear(2000, num_features)
        self.relu = nn.ReLU()
        self.fc_d1 = nn.Linear(500, dimin)
        self.fc_d2 = nn.Linear(500, 500)
        self.fc_d3 = nn.Linear(2000, 500)
        self.fc_d4 = nn.Linear(num_features, 2000)
        self.pretrainMode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def setP(self, p):
        self.dropout.p = p

    def setPretrain(self, mode):
        """To set training mode to pretrain or not, 
        so that it can control to run only the Encoder or Encoder+Decoder"""
        self.pretrainMode = mode

    def forward1(self, x):
        x_ae = x
        x = x.view(-1, 1 * 28 * 28)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.dropout(x)
        x = self.fc_d1(x)
        x_de = x.view(-1, 1, 28, 28)
        return x_ae, x_de

    def forward2(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x_ae = x
        #
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        #
        x = self.dropout(x)
        x = self.fc_d2(x)
        x = self.relu(x)
        x_de = x
        return x_ae, x_de

    def forward3(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.fc2(x)
        x = self.relu(x)
        x_ae = x
        #
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        #
        x = self.dropout(x)
        x = self.fc_d3(x)
        x = self.relu(x)
        x_de = x
        return x_ae, x_de

    def forward4(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.fc2(x)
        x = self.relu(x)
        #
        x = self.fc3(x)
        x = self.relu(x)
        x_ae = x
        #
        x = self.dropout(x)
        x = self.fc4(x)
        #
        x = self.dropout(x)
        x = self.fc_d4(x)
        x = self.relu(x)
        x_de = x
        return x_ae, x_de

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.fc2(x)
        x = self.relu(x)
        #
        x = self.fc3(x)
        x = self.relu(x)
        #
        x = self.fc4(x)
        x_ae = x
        # if not in pretrain mode, we only need encoder
        if self.pretrainMode == False:
            return x
        #
        x = self.fc_d4(x)
        x = self.relu(x)
        #
        x = self.fc_d3(x)
        x = self.relu(x)
        #
        x = self.fc_d2(x)
        x = self.relu(x)
        #
        x = self.fc_d1(x)
        x_de = x.view(-1, 1, 28, 28)
        return x_ae, x_de

    def decode(self, x):
        x = self.fc_d4(x)
        x = self.relu(x)
        #
        x = self.fc_d3(x)
        x = self.relu(x)
        #
        x = self.fc_d2(x)
        x = self.relu(x)
        #
        x = self.fc_d1(x)
        x_de = x.view(-1, 1, 28, 28)
        return x_de


class AECIFAR10(nn.Module):
    def __init__(self):
        super(AECIFAR10, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(48, 48, 4, stride=2),  # [batch, 48, 1, 1]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 48, 4, stride=2),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

if __name__ == '__main__':
    pass
