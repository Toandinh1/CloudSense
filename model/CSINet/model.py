#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:30:23 2024

@author: jackson-devworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import HumanPoseEstimator

def compute_compression_rate(original_tensor: torch.Tensor, compressed_tensor: torch.Tensor):
    """
    Compute the compression rate of a tensor.

    Parameters:
    - original_tensor: The original tensor before compression.
    - compressed_tensor: The tensor after compression.

    Returns:
    - A tensor representing the compression rate.
    """
    #print(original_tensor.size(), compressed_tensor.size())
    # Get the sizes of the original and compressed tensors
    original_size = original_tensor.numel()  # Total number of elements in the original tensor
    compressed_size = compressed_tensor.numel()  # Total number of elements in the compressed tensor

    # Compute compression rate
    compression_rate = original_size / compressed_size if compressed_size > 0 else torch.tensor(float('inf'))  # Handle division by zero

    return compression_rate

class CsiNetAutoencoder(nn.Module):
    def __init__(self, config, check_compression_rate, compression_rate):
        super(CsiNetAutoencoder, self).__init__()
        self.check_compression_rate = check_compression_rate
        self.img_channels = config["img_channels"]
        self.img_height = config["img_height"]
        self.img_width = config["img_width"]
        self.residual_num = config["residual_num"]
        self.img_total = self.img_channels * self.img_height * self.img_width
        self.encoded_dim = int(self.img_total / compression_rate)  # Adjust encoded dimension based on compression rate
        #print(self.encoded_dim)
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=3, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(3)
        self.leaky_relu1 = nn.LeakyReLU()

        # Encoder
        self.flatten = nn.Flatten()
        self.encoder_dense = nn.Linear(3420, self.encoded_dim)

        # HPE task
        self.human_pose_estimator = HumanPoseEstimator(self.encoded_dim, 34, 32)
        
        # Decoder
        self.decoder_dense = nn.Linear(self.encoded_dim, self.img_total)
        self.unflatten = nn.Unflatten(1, (3, self.img_height, self.img_width))

        # Residual blocks in the decoder
        self.residual_blocks = nn.ModuleList([
            self._residual_block() for _ in range(self.residual_num)
        ])

        # Final convolutional layer
        self.conv_out = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def _residual_block(self):
            return nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3)
            )
    def forward(self, x):
        batch = x.size(0)
        
        # Initial convolutional layer
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)

        # Encoder
        x = self.flatten(x)
        encoded = self.encoder_dense(x)
        
        # Optional: Check compression rate
        if self.check_compression_rate:
            with torch.no_grad():
                compression_rate = compute_compression_rate(x, encoded)
            print(f"Compression rate: {compression_rate}")
        
        # Decoder
        r_x = self.decoder_dense(encoded)
        r_x = self.unflatten(r_x)

        # Residual blocks
        for block in self.residual_blocks:
            residual = r_x
            r_x = block(r_x)
            r_x += residual
            r_x = self.leaky_relu1(r_x)

        # Final convolutional layer
        r_x = self.conv_out(r_x)
        r_x = self.sigmoid(r_x)
        #r_x = F.interpolate(r_x, size=(114, 10), mode='bilinear', align_corners=False)

        # HPE task
        pred_keypoint = self.human_pose_estimator(encoded).reshape(batch, 17, 2)
        
        return r_x, pred_keypoint