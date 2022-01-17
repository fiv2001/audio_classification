import PIL
import io

import pandas as pd
import numpy as np

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import torchaudio
from torchvision.transforms import ToTensor
from torchvision import models

from dataset import UrbanSoundDataset
from train import train, test
from model import get_model, get_path
from util import create_config


path_to_UrbanSound8K_csv = Path('/home/pinkbittern/audio_classification/data/UrbanSound8K.csv')
path_to_UrbanSound8K_audio = Path('/home/pinkbittern/audio_classification/data')

print("Welcome to audio classification")

config = create_config()

train_set = UrbanSoundDataset(path_to_UrbanSound8K_csv, path_to_UrbanSound8K_audio, range(1,10), hop_length=config.getint('hop_length'), number_of_mel_filters=config.getint('number_of_mel_filters'),
                              resample_freq=config.getint('resample_freq'), return_audio=False)
test_set = UrbanSoundDataset(path_to_UrbanSound8K_csv, path_to_UrbanSound8K_audio, [10], hop_length=config.getint('hop_length'), number_of_mel_filters=config.getint('number_of_mel_filters'),
                             resample_freq=config.getint('resample_freq'), return_audio=True)
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

train_loader = torch.utils.data.DataLoader(train_set, batch_size = config.getint('batch_size'), shuffle = True, pin_memory=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = config.getint('batch_size'), shuffle = False, pin_memory=False, num_workers=1)

model = get_model(config)

optimizer = optim.SGD(model.parameters(), lr = config.getfloat('base_lr'), momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.getint('number_of_epochs')//3, gamma = 0.3)
criterion = nn.CrossEntropyLoss()

device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

print('Device to use: {}'.format(device))

for epoch in range(config.getint('number_of_epochs')):
    train(model, epoch, train_loader, device, optimizer, criterion, config.getint('log_interval'))
    test(model, epoch, test_loader, device, config)
    scheduler.step()
    torch.save(model.state_dict(), get_path(epoch + 1)) 
