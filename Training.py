#this is a somewhat copy of the original training script from the 'price-prediction' repo i made but refactored for images
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import kagglehub
import numpy as np