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

from pathlib import Path

# File Path for images
image_dir = Path("/Users/ayush/Ayush/Projects/predage/20-50")

filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(int)
# Dataframe with all our image and their labels
raw_df = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

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

class AgePredictionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx, 1] - 20

        if self.transform:
            image = self.transform(image)

        return image, label

# Define any transformations if needed
# Here are the stats (Mean and std dev for each channel (RGB))
stats = ((0.4509, 0.3791, 0.3385), (0.2023, 0.1994, 0.2010))

# Train transformations
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), # Randomly applies padding of 4 and reflects
                         tt.RandomHorizontalFlip(), # Randomly flips some of the images horizontally
                          tt.RandomRotation(15),
                          tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
                          tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), # Convert it to tensors
                         tt.Normalize(*stats,inplace=True)]) # Normalize the pixels

# Validation transformations
# We do not add any random transformations as we should not mess with images when trying on test datasets
valid_tfms = tt.Compose([
    tt.Resize(256),  # Ensures validation images are resized to 256x256
    tt.ToTensor(),  # Converts images to tensors
    tt.Normalize(*stats)  # Normalizes the images
])


train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_ds = AgePredictionDataset(train_df, transform=train_tfms)
val_ds = AgePredictionDataset(val_df, transform=valid_tfms)


# Creating Dataloaders


from torch.utils.data import random_split
random_seed = 42
torch.manual_seed(random_seed)
# Split dataset between training and validation sets

from torch.utils.data import DataLoader

# Create batch sizes so it is easier to train models
batch_size = 128

# train_dl contains all the training data we need (feature,label)
train_dl = DataLoader(train_ds, batch_size,num_workers=4,pin_memory = True ,shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2,num_workers=4,pin_memory = True )

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


def get_loaders():
    return train_dl, val_dl

print(len(raw_df))
print(len(train_ds))