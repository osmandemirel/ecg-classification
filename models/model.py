import torch
from torch import nn
import torch.nn.functional as F

"""
Following methods are created for ease of use.
    conv3x3: 2D convolutional layer with kernel_size=3x3, stride=1, padding=1
    conv1x1: 2D convolutional layer with kernel_size=1x1, stride=1
    conv2d_block: combination of following:
        Conv2d
        ReLU
        BatchNorm(optional. set via param 'bn'
"""
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(kernel_size=3,in_channels=in_channels, out_channels=out_channels, stride=1,padding=(1,1))

def conv1x1(in_channels,out_channels):
    return nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1)

def conv2d_block(in_channels,out_channels,kernel_size=3,stride=1,padding=(1,1),bn=False):
    layers = []
    layers.append(nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,out_channels=out_channels, stride=stride,padding=padding))
    if bn:
        layers.append(nn.BatchNorm2d(num_features=out_channels))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class BottleNeck(nn.Module):
    """
    Bottleneck is essentially, a 2 times 3x3 conv2d blocks replaced by 1x1,3x3,1x1 blocks
    """
    def __init__(self, num_channels, width):
        # first 1x1 block
        super().__init__()
        self.conv1x1_1 = conv1x1(in_channels=num_channels, out_channels=width)
        self.bn1 = nn.BatchNorm2d(num_features=width)
        self.relu1 = nn.ReLU()

        # 3x3 block
        self.conv3x3 = conv3x3(width,width)
        self.bn2 = nn.BatchNorm2d(num_features=width)
        self.relu2 = nn.ReLU()

        # second 1x1 block
        self.conv1x1_2 = conv1x1(in_channels=width, out_channels=num_channels)
        self.bn3 = nn.BatchNorm2d(num_features=num_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # save input for residual block
        inp = x
        # forward through first 1x1 block
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # forward through 3x3 block
        x = self.conv3x3(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # forward through second 1x1 block
        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x+=inp
        out = self.relu3(x)
        return out

class ECGHeartbeat(nn.Module):
    """
    Based on architecture provided by following paper:
        ECG Heartbeat Classification: A Deep Transferable Representation
        Mohammad Kachuee, Shayan Fazeli, Majid Sarrafzadeh
    """
    def __init__(self):
        super(ECGHeartbeat, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=32,out_channels=256,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(in_features=256*6*5,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=27),
            nn.Sigmoid()
            )

    def forward(self,x):
        out = self.net(x)
        return out


class ArrhythmiaNet(nn.Module):
    """
    Based on architecture provided by following paper:
        ECG arrhythmia classification using a 2-D convolutional neural network
        Tae Joon Jun1, Hoang Minh Nguyen1,Daeyoun Kang1,Dohyeun Kim1,Daeyoung Kim1,Young-Hak Kim2
    """
    def __init__(self):
        super(ArrhythmiaNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(in_features=256*14*12,out_features=2048),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,out_features=27),
            nn.Softmax()
        )

    def forward(self,x):
        out = self.net(x)
        return out

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
