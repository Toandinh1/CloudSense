#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:36:14 2024

@author: jackson-devworks
"""
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, compress_rate):
        super().__init__()
        self.compress_rate = compress_rate
        self.encoded_dim = (3 * 114 * 10) // self.compress_rate  # Calculate encoded dimension
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.fc = nn.Linear(in_features=2 * 114 * 10, out_features=self.encoded_dim)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv_block(x)
        out = torch.reshape(out, (batch_size, -1))  # Reshape to (batch_size, 2*Nc*Nt)
        out = self.fc(out)  # Get encoded representation
        return out


class Decoder(nn.Module):
    def __init__(self, compress_rate):
        super().__init__()
        self.compress_rate = compress_rate
        self.encoded_dim = (3 * 114 * 10) // self.compress_rate  # Calculate encoded dimension
        self.fc = nn.Linear(in_features=self.encoded_dim, out_features=3 * 114 * 10)  # Match output size
        self.refine1 = Refinenet()
        self.refine2 = Refinenet()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.fc(x)
        out = torch.reshape(out, (batch_size, 3, 114, 10))  # Reshape to match input shape
        out = self.refine1(out)
        out = self.refine2(out)
        return out

class Refinenet(nn.Module):
    # input: (batch_size, 2, Nc, Nt)
    # output: (batch_size, 2, Nc, Nt)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=3),
            nn.LeakyReLU(negative_slope=0.3),
        )

    def forward(self, x):
        skip_connection = x
        out = self.conv1(x)
        # out.shape = (batch_size, 8, Nc, Nt)
        out = self.conv2(out)
        # out.shape = (batch_size, 16, Nc, Nt)
        out = self.conv3(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = out + skip_connection

        return out
        
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