import torch
from torch import nn



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sideway=False):
        super().__init__()

        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),

            nn.Conv2d(out_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels)
        )

        if sideway:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Sequential()

    def forward(self, x):

        res = self.skip(x)
        out = self.main(x)

        return out + res




class Resnet(nn.Module):
    def __init__(self, builder, nums):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
 
        self.pool = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.block1 = self._create_layer_(builder, 64, 64, 1, nums[0])
        self.block2 = self._create_layer_(builder, 64, 128, 2, nums[1])
        self.block3 = self._create_layer_(builder, 128, 256, 2, nums[2])
        self.block4 = self._create_layer_(builder, 256, 512, 2, nums[3])

        self.bn = nn.BatchNorm2d(512)

        self.fc = nn.Sequential(
            nn.Linear(512*7*6, 512),
            nn.BatchNorm1d(512)
        )

    def _create_layer_(self, builder, in_channels, out_channels, stride=1, num=1):
        
        layers = list()
        layers.append(builder(in_channels, out_channels, stride, True))
        for _ in range(num-1):
            layers.append(builder(out_channels, out_channels, 1, False))
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.bn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def _resnet(builder, nums):

    model = Resnet(builder, nums)

    return model


def resnet18():
    
    return _resnet(Block, [2, 2, 2, 2])


def resnet34():

    return _resnet(Block, [3, 5, 4, 3])


def resnet50():
    
    return _resnet(Block, [3, 13, 4, 3])


def resnet101():

    return _resnet(Block, [3, 13, 30, 3])