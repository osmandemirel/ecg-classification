import torch
from torch import nn
import torch.nn.functional as F

def conv1d_block(in_channels,out_channels, bn=False):
    """
    :param in_channels: number of input channels for conv net, int
    :param out_channels: number of output channels for conv net, int
    :param bn: batch normalization flag, boolean. Adds a batch norm layer between conv and Relu if bn is set to True
    :return: Sequential layers, sub-network consists of conv bn relu
    """
    layers = []
    layers.append(nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=50,stride=1,padding=24))
    if bn:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ECGNet(nn.Module):
    """
    Feature extractor module similar to what is used in VGG16
    """
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.conv_block_1 = conv1d_block(in_channels=in_channels,out_channels=64,bn=True)
        self.conv_block_2 = conv1d_block(in_channels=64, out_channels=2, bn=True)
        self.conv_layer = nn.Conv1d(in_channels=2,out_channels=2,kernel_size=3,stride=1,padding=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1*5000*2,out_features=num_classes)
        self.sfmx = nn.Softmax()

    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        out = self.sfmx(out)
        return out
