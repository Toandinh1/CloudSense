#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:29:52 2024

@author: jackson-devworks
"""
from torch import nn
import torch
import torch.nn.functional as F

from .module import FeatureEncoder, FeatureDecoder, HumanPoseEstimator

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

class DeepCMC(nn.Module):
    def __init__(self):
        super(DeepCMC, self).__init__()
        self.encoder = FeatureEncoder()
        self.decoder = FeatureDecoder()
        self.human_pose_estimator = HumanPoseEstimator(1024, 34, 32)

    def forward(self, x, check_compression_rate = False):
        batch = x.shape[0]
        quantized, indices = self.encoder(x)
        if check_compression_rate:
            with torch.no_grad():
                compression_rate = compute_compression_rate(x, quantized)
            
            print(f"\nEncoder size: {quantized.size()}")
            print(f"Compression rate: {compression_rate}")
        decoded = self.decoder(quantized)
        red_keypoint = self.human_pose_estimator(quantized).reshape(batch, 17, 2)
        r_x = F.interpolate(decoded, size=(114,10), mode="bilinear", align_corners=True)
        return decoded, red_keypoint