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


class FeatureExtractor(nn.Module):
    """
    Feature extractor module similar to what is used in VGG16
    """
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_block_1 = conv1d_block(in_channels=in_channels,out_channels=64,bn=True)
        self.conv_block_2 = conv1d_block(in_channels=64, out_channels=out_channels, bn=True)
        self.conv_layer = nn.Conv1d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=2)

    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_layer(x)
        return x


class FeatureClassifier(nn.Module):
    """
    Takes vector inputs and returns class probabilities
    """
    def __init__(self,in_features,num_classes):
        super().__init__()
        self.fc_1 = nn.Linear(in_features=in_features,out_features=in_features//10)
        self.relu_1 = nn.ReLU()
        self.fc_3 = nn.Linear(in_features=in_features//10,out_features=num_classes)
        self.s_max = nn.Softmax()

    def forward(self,x):
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_3(x)
        out = self.s_max(x)
        return out


class ECGNet(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, num_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels, out_channels)
        self.flatten = nn.Flatten()
        self.feature_classifier = FeatureClassifier(in_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        out = self.feature_classifier(x)
        return out
