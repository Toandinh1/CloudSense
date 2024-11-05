#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:29:30 2024

@author: jackson-devworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = ConvBlock(channels, channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class Quantize(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Quantize, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Initialize the embedding table for quantization
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Flatten input to (batch * height * width, embedding_dim)
        x_flattened = x.view(-1, self.embedding_dim)

        # Compute distances between x and embeddings
        distances = (
            torch.sum(x_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(x_flattened, self.embeddings.weight.t())
        )

        # Get the closest embedding for each vector in x
        indices = torch.argmin(distances, dim=1)

        # Reshape indices back to the original spatial dimensions
        indices = indices.view(x.shape[0], *x.shape[2:])  # Restore batch and spatial dimensions

        # Quantize using the closest embeddings
        quantized = self.embeddings(indices).view_as(x)  # Shape should match x

        return quantized, indices
    
class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.conv1 = ConvBlock(3, 32, kernel_size=9, stride=4, padding=4)
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=5, stride=2, padding=2)
        self.quantize = Quantize(num_embeddings=512, embedding_dim=128)  # Example parameters

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Quantize the output
        quantized, indices = self.quantize(x)
        
        # Placeholder for entropy encoding (handled outside the model in practice)
        return quantized, indices

class FeatureDecoder(nn.Module):
    def __init__(self):
        super(FeatureDecoder, self).__init__()
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)
        self.res_block3 = ResidualBlock(128)
        self.conv1 = ConvBlock(128, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = ConvBlock(64, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=9, stride=4, padding=4)  # Output layer without BN and ReLU

    def forward(self, quantized):
        # Placeholder for entropy decoding (handled outside the model in practice)
        
        x = self.res_block1(quantized)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    
class HumanPoseEstimator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(HumanPoseEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim*2)
        self.relu = nn.ReLU()  # Hàm kích hoạt ReLU
        self.dropout = nn.Dropout(p=0.1) 
        self.gelu = nn.GELU()
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        #x= x.reshape(x.size(0), x.size(1)*x.size(2))
        
        x = self.fc1(x)
        x = self.relu(x)  # Áp dụng ReLU sau lớp fully connected thứ nhất
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.relu(x)  # Áp dụng ReLU sau lớp fully connected thứ hai
        x = self.dropout(x)
        output = self.fc3(x)

        return output