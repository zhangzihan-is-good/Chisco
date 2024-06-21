# MIT License
# 
# Copyright (c) 2024 Zihan Zhang, Yi Zhao, Harbin Institute of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
import math

class EEGcnn(nn.Module):
    def __init__(self, Chans=64, dropoutRate=0.5, kernLength1=100, kernLength2=50, F1=8, D=2, F2=16, P1=2, P2=5, dropoutType='Dropout'):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding='same', bias=False), # (N, F1, Chans, Samples)
            nn.BatchNorm2d(F1), # (N, F1, Chans, Samples)
            nn.Conv2d(F1, D*F1, (Chans, 1), groups=F1, bias=False), # (N, D*F1, 1, Samples)
            nn.BatchNorm2d(D*F1), # (N, D*F1, 1, Samples)
            nn.ELU(), # (N, D*F1, 1, Samples)
            nn.AvgPool2d((1, P1)), # (N, D*F1, 1, Samples//2)
            nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate) # (N, D*F1, 1, Samples//2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, kernLength2), groups=D*F1, padding='same', bias=False), # (N, D*F1, 1, Samples//2)
            nn.Conv2d(D*F1, F2, (1, 1), bias=False), # (N, F2, 1, Samples//2)
            nn.BatchNorm2d(F2), # (N, F2, 1, Samples//2)
            nn.ELU(), # (N, F2, 1, Samples//2)
            nn.AvgPool2d((1, P2)), # (N, F2, 1, Samples//4)
            nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate) # (N, F2, 1, Samples//4)
        )

    def forward(self, input): # (N, Chans, Samples)
        input = torch.unsqueeze(input, dim=1) # (N, 1, Chans, Samples)
        input = self.block1(input)
        input = self.block2(input)
        input = torch.squeeze(input, dim=2) # (N, F2, Samples//4)
        return input

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)