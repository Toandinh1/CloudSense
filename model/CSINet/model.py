#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:30:23 2024

@author: jackson-devworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import HumanPoseEstimator, Encoder, Decoder

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

input_dim_hpe = {
    1: 3420,
    34: 100,
    84: 40,
    1710: 2
}
class CsiNetAutoencoder(nn.Module):
    def __init__(self, config, compression_rate):
        super(CsiNetAutoencoder, self).__init__()
        self.encoder = Encoder(compression_rate)
        self.decoder = Decoder(compression_rate)
        self.human_pose_estimator = HumanPoseEstimator(input_dim_hpe[compression_rate], 34, 32)
    def forward(self, x, check_compression_rate=False):
        batch = x.size(0)
        
        # Initial convolutional layer
        
        encoded = self.encoder(x)
        # Optional: Check compression rate
        if check_compression_rate:
            with torch.no_grad():
                compression_rate = compute_compression_rate(x, encoded)
            
            print(f"\nEncoder size: {encoded.size()}")
            print(f"Compression rate: {compression_rate}")
        
        # Decoder
        r_x = self.decoder(encoded)

        # HPE task
        pred_keypoint = self.human_pose_estimator(encoded).reshape(batch, 17, 2)
        
        return r_x, pred_keypoint