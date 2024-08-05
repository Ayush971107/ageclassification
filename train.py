from data import get_loaders
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F



def get_default_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



# Training Phase

from torchvision import models
import torch.nn as nn
import torch.optim as optim

# Load the pretrained ResNet18 model





if __name__ == "__main__":
    train_dl, val_dl = get_loaders()
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    model3 = models.resnet18(pretrained=True)

# Modify the final fully connected layer to output 31 classes
    num_ftrs = model3.fc.in_features
    model3.fc = nn.Linear(num_ftrs, 31)  # Assuming 31 output classes

# Move the model to the appropriate device
    model3 = to_device(model3, device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model3.parameters(), lr=0.01, weight_decay=1e-3)

    print(train_dl.device)

    for epoch in range(5):
        running_loss = 0.0
        for images, labels in train_dl:
            images, labels = to_device(images, device), to_device(labels, device)  # Ensure data is on the correct device

            optimizer.zero_grad()
            outputs = model3(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("batch done")
        epoch_loss = running_loss / len(train_dl)
        print(epoch_loss)

    print("Training completed.")

