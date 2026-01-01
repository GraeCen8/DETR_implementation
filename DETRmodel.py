#this is to define the model architecture

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

# Define the DETR model class
class DETR(nn.Module):
    def __init__(self, num_classes, num_queries, t_d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()
        # Load a pre-trained ResNet50 model
        self.backbone = models.resnet50(pretrained=True) #eventually will replace with my custom model
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the classification head
 
        # Define the Transformer
        self.transformer = Transformer(d_model=t_d_model,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        activation=nn.GELU(),
                                        dropout=0.1
                                        )
                                              

        # Define the input projection layer
        self.input_proj = nn.Conv2d(2048, t_d_model, kernel_size=1)

        # Define the object query embeddings
        self.query_embed = nn.Embedding(num_queries, t_d_model)

        # Define the classification and bounding box prediction heads
        self.class_embed = nn.Sequential(
            nn.Linear(t_d_model, t_d_model),
            nn.ReLU(),
            nn.Linear(t_d_model, t_d_model),
            nn.ReLU(),
            nn.Linear(t_d_model, num_classes + 1)  # +1 for the no-object class
        )
        self.bbox_embed = nn.Sequential(
            nn.Linear(t_d_model, t_d_model),
            nn.ReLU(),
            nn.Linear(t_d_model, t_d_model),
            nn.ReLU(),
            nn.Linear(t_d_model, 4)
        )
        )
        self.bbox_embed = nn.Sequential(
            nn.Linear(t_d_model, t_d_model),
            nn.ReLU(),
            nn.Linear(t_d_model, t_d_model),
            nn.ReLU(),
            nn.Linear(t_d_model, 4)
        )
    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        features = self.input_proj(features).flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        # Prepare object queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)  # (num_queries, B, C)

        # Pass through the Transformer
        transformer_output = self.transformer(features, query_embed)

        # Predict classes and bounding boxes
        class_logits = self.class_embed(transformer_output)
        bbox_preds = self.bbox_embed(transformer_output).sigmoid()

        return class_logits, bbox_preds