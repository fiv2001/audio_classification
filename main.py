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

class UrbanSoundDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, file_path, folderList, hop_length, resample_freq=0, return_audio=False):
        self.file_path = file_path
        self.file_names = []
        self.labels = []
        self.folders = []
        self.n_mels = number_of_mel_filters
        self.resample = resample_freq
        self.return_audio = return_audio
        self.hop_length = hop_length
        
        #loop through the csv files and only add those from the folder list
        csvData = pd.read_csv(csv_path)
        for i in range(0,len(csvData)):
            if csvData.iloc[i, 5] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])       

    def __getitem__(self, index):
        #format the file path and load the file
        path = self.file_path / ("fold" + str(self.folders[index])) / self.file_names[index]
        soundData, sample_rate = torchaudio.load(path, normalize=True)

        if self.resample > 0:
#            print(soundData.shape)
            resample_transform = torchaudio.transforms.Resample(
              orig_freq=sample_rate, new_freq=self.resample)
            soundData = resample_transform(soundData)
#            print(soundData.shape)

        # This will convert audio files with two channels into one
        soundData = torch.mean(soundData, dim=0, keepdim=True)
        if index == 0:
            print(soundData.shape)
        # Convert audio to log-scale Mel spectrogram
        melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.resample,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        melspectrogram = melspectrogram_transform(soundData)
        melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

        #Make sure all spectrograms are the same size
        fixed_length = 3 * (self.resample) // self.hop_length
        if melspectogram_db.shape[2] < fixed_length:
            melspectogram_db = torch.nn.functional.pad(melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
        else:
            melspectogram_db = melspectogram_db[:, :, :fixed_length]

        if index == 0:
            print(melspectogram_db.shape)
        return self.resample, melspectogram_db, self.labels[index]

    def __len__(self):
        return len(self.file_names)

def train(model, epoch):
    model.train()
    for batch_idx, (sample_rate, inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iteration = epoch * len(train_loader) + batch_idx
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss))
            print('loss, iteration: ', loss, iteration)
            print('learning rate: , ', optimizer.param_groups[0]['lr'], iteration)


def test(model, epoch):
    model.eval()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for idx, (sample_rate, inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(inputs)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
        
            iteration = (epoch + 1) * len(train_loader)

    total_accuracy = 100 * sum(class_correct)/sum(class_total)
    print('[Iteration {}] Accuracy on the {} test images: {}%\n'.format(epoch, sum(class_total), total_accuracy))
    print('accuracy, iteration: ', total_accuracy, iteration)

path_to_UrbanSound8K_csv = Path('/home/pinkbittern/audio_classification/data/UrbanSound8K.csv')
path_to_UrbanSound8K_audio = Path('/home/pinkbittern/audio_classification/data')
print("HELLO")

resample_freq = 22050
number_of_mel_filters = 64
dropout = 0.25
base_lr = 0.005
number_of_epochs = 6
batch_size = 15
log_interval = 10
debug_interval = 10
start_epoch = 0
end_epoch = 10
hop_length = 100

train_set = UrbanSoundDataset(path_to_UrbanSound8K_csv, path_to_UrbanSound8K_audio, range(1,10), hop_length=hop_length, 
                              resample_freq=resample_freq, return_audio=False)
test_set = UrbanSoundDataset(path_to_UrbanSound8K_csv, path_to_UrbanSound8K_audio, [10], hop_length=hop_length,
                             resample_freq=resample_freq, return_audio=True)
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False, pin_memory=False, num_workers=1)

classes = ('air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
           'gun_shot', 'jackhammer', 'siren', 'street_music')

model = models.resnet18(pretrained=True)
model.conv1=nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size[0], 
                      stride=model.conv1.stride[0], padding=model.conv1.padding[0])
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(*[nn.Dropout(p=dropout), nn.Linear(num_ftrs, len(classes))])

def get_path(epoch_number):
    return Path(f'/home/pinkbittern/audio_classification/checkpoints/{start_epoch}.pth')

if start_epoch != 0:
    path = get_path(start_epoch) 
    model.load_state_dict(torch.load(path))

optimizer = optim.SGD(model.parameters(), lr = base_lr, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = number_of_epochs//3, gamma = 0.1)
criterion = nn.CrossEntropyLoss()

device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print('Device to use: {}'.format(device))

for epoch in range(number_of_epochs):
    train(model, epoch)
    test(model, epoch)
    scheduler.step()
    torch.save(model.state_dict(), get_path(epoch + 1)) 
