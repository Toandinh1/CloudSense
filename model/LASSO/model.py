#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:03:24 2024

@author: jackson-devworks
"""
import numpy as np
from torch import nn
import torch

from .module import lasso_compress, HumanPoseEstimator

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


compress_dict = {
    1: {
        "alpha": 0.1,
        "encoded_dim": 128,
        "input_hpe": 3420
    }
}
class LassoAutoEncoder(nn.Module):
    def __init__(self, config, compression_rate):
        super(LassoAutoEncoder, self).__init__()
        self.input_shape = config["input_shape"]
        self.flat_input_size = np.prod(self.input_shape)
        self.alpha = compress_dict[compression_rate]["alpha"]

        
        self.decoder = nn.Sequential(
            nn.Linear(compress_dict[compression_rate]["input_hpe"], self.flat_input_size),
            nn.ReLU(),
            nn.Unflatten(1, self.input_shape)
        )
        
        self.human_pose_estimator = HumanPoseEstimator(compress_dict[compression_rate]["input_hpe"], 34, 32)
        
    def forward(self, x, check_compression_rate = False):
        batch = x.size(0)
        
        # Step 2: LASSO encoder (compress the input)
        compressed_x = lasso_compress(x)
        if check_compression_rate:
            with torch.no_grad():
                compression_rate = compute_compression_rate(x, compressed_x)
            
            print(f"\nEncoder size: {compressed_x.size()}")
            print(f"Compression rate: {compression_rate}")
        
        reconstructed_x = self.decoder(compressed_x)
        pred_keypoint = self.human_pose_estimator(compressed_x).reshape(batch, 17, 2)
        return reconstructed_x, pred_keypoint
