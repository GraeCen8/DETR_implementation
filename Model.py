
#imports
import torch as t
import torch.nn as nn
import torchvision.models as models
from torch.nn import Transformer
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import math
from PIL import Image
import numpy as np
import requests
import pandas as pd
from io import BytesIO
from kagglehub import Dataset

class DoubleConv(nn.Module)
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super(Unet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Downward section
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #up part
        for feature in reversed(feature):
            self.ups.append(
            nn.ConvTranspose2d(
                feature*2,
                feature
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        #bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.end_conv = nn.Conv2d(features[0], out_channels, kernal_size=1)
    
    def forward(self, x):
        skip_connections = []

    #ups
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

    #down
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connections = skip_connections[idx//2]
            if x.shape != skip_connections.shape:
                x = TF.resize(x,size=skip_connection.shape[2:1])
            concat_skip = torch.cat((skip_connections, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        x = self.end_conv(x)
        return x 