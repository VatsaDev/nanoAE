# basic AE, image in/out, compression

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):

    def __init__(self, in_f):
        super().__init__()

        self.into = nn.Linear(in_f, 4*in_f)
        self.silu = nn.SiLU(inplace=True)
        self.out = nn.Linear(4*in_f, in_f)

    def forward(self, x):

        B, C, H, W = x.shape

        x = x.view(B, H, W, C)
        
        x = self.into(x) # channel wise
        x = self.silu(x)
        x = self.out(x)

        x = x.view(B, C, H, W)

        return x
        
class Auto_Encoder(nn.Module):

    def __init__(self, input_size, latent_size):
        super(Auto_Encoder, self).__init__()
        
        nc = 256
        
        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, nc, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(1, nc),
            MLP(nc),            
            nn.SiLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((latent_size, latent_size)),
            
            nn.Conv2d(nc, nc//4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, nc//4),
            nn.SiLU(inplace=True),
            MLP(nc//4),
            
            nn.Conv2d(nc//4, 3, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(3, nc//4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, nc//4),
            nn.SiLU(inplace=True),
            MLP(nc//4),
            
            nn.Conv2d(nc//4, nc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, nc),
            nn.SiLU(inplace=True),
            MLP(nc),
            
            nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True),
            nn.Conv2d(nc, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x): # output image and latent
        
        encoded = self.enc(x)
        decoded = self.dec(encoded)

        return encoded, decoded
