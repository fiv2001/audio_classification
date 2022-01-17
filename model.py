import torch
import torch.nn as nn

from torchvision import models
from pathlib import Path


def get_model(config):
    model = models.resnet18(pretrained=True)
    model.conv1=nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size[0],
                          stride=model.conv1.stride[0], padding=model.conv1.padding[0])
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(*[nn.Dropout(p=config.getfloat('dropout')), nn.Linear(num_ftrs, config.getint('classes_num'))])

    start_epoch = config.getint('start_epoch')
    if start_epoch != 0:
        path = get_path(start_epoch)
        model.load_state_dict(torch.load(path))

    return model

def get_path(epoch_number):
    return Path(f'/home/pinkbittern/audio_classification/checkpoints/{epoch_number}.pth')


