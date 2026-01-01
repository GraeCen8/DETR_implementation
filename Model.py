
#imports
import torch as t
import torch.nn as nn
from torch.nn import Transformer
import math
import numpy as np
import pandas as pd

class DoubleConv(nn.Module):
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

        #upward section
        for feature in reversed(features):
            self.ups.append(
            nn.ConvTranspose2d(
                feature*2,
                feature,
                kernel_size=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        #bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.end_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
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
            concat_skip = torch.cat((skip_connections, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        x = self.end_conv(x)
        return x 
    

def test(x_pxls, y_pxls):
    x = t.randn(3, 1, x_pxls, y_pxls)
    model = Unet(in_channels=1, out_channels=1)
    preds = model(x)

    if x.shape == preds.shape:
        print(f'test for Tensor size errors is a pass with {[x_pxls, y_pxls]}')
    else:
        print(f'test for tensor size errors is a fail with {[x_pxls, y_pxls]}')

    
if __name__ == '__main__':
    test(160, 160)
    test(161,161)
    test(160,320)
