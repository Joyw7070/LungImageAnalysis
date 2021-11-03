''''
resnet18
kernel_size=3
'''

from torch import nn
import torch as t
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    '''定义实线部分的残差块，通道数相同'''
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=3, stride=1, padding = 1,bias=False)
        self.bn = nn.BatchNorm2d(num_features=in_channels, eps=1e-05, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,bias=False)
    def forward(self,x):
        identity = x
        y = self.bn(self.conv1(x))
        y = F.relu(y)
        y = self.bn(self.conv2(y))
        return F.relu(y+identity)

class ResidualBlock_plus(nn.Module):
    '''定义虚线部分的残差块，通道数不同，'''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_plus, self).__init__()
        self.downsample = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size=1, stride=2,padding=0, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True)
        self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 3,stride=2, padding= 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        identity = self.downsample(x)
        identity = self.bn(identity)
        y = self.conv1(x)
        y = self.bn(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn(y)
        return F.relu(y+identity)


class myResNet18(nn.Module):
    def __init__(self, num_classes, drop_prob):
        super(myResNet18, self).__init__()
        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.resn1 = ResidualBlock(64)
        self.resn2 = ResidualBlock(64)
        self.resn3 = ResidualBlock_plus(64,128)
        self.resn4 = ResidualBlock(128)
        self.resn5 = ResidualBlock_plus(128, 256)
        self.resn6 = ResidualBlock(256)
        self.resn7 = ResidualBlock_plus(256, 512)
        self.resn8 = ResidualBlock(512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features= 1024),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(p=drop_prob)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self,x):
        in_size = x.size(0)
        x = self.preconv(x)
        x = self.resn1(x)
        x = self.resn2(x)
        x = self.resn3(x)
        x = self.resn4(x)
        x = self.resn4(x)
        x = self.resn5(x)
        x = self.resn6(x)
        x = self.resn6(x)
        x = self.resn7(x)
        x = self.resn8(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return F.softmax(out, dim=1)


