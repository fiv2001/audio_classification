import pandas as pd
import numpy as np

from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

class UrbanSoundDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, file_path, folderList, hop_length, number_of_mel_filters=64, resample_freq=0, return_audio=False):
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
