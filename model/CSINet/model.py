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
    4: 855,
    16: 213,
    32: 106,
    64: 53
}

def quantize_and_restore(tensor):
    """
    Quantizes a tensor by scaling, converting to integers, and then restores it to float32.

    Parameters:
    - tensor (torch.Tensor): The input tensor to quantize.
    - scale_factor (float): A scaling factor to minimize information loss during conversion.

    Returns:
    - torch.Tensor: The tensor restored to float32 with minimized quantization loss.
    """    
    # Step 2: Convert the scaled tensor to integers (simulate quantization)
    int_tensor = tensor.to(torch.int32)
    
    # Step 3: Convert back to float and undo scaling
    restored_tensor = int_tensor.to(torch.float32)
    
    return restored_tensor

class CsiNetAutoencoder(nn.Module):
    def __init__(self, config, compression_rate, output_hpe = 36):
        super(CsiNetAutoencoder, self).__init__()
        self.encoder = Encoder(compression_rate)
        self.decoder = Decoder(compression_rate)
        self.human_pose_estimator = HumanPoseEstimator(input_dim_hpe[compression_rate], output_hpe, 32)
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
        pred_keypoint = self.human_pose_estimator(encoded).reshape(batch, -1, 2)
        
        return r_x, pred_keypoint
    

if __name__ == "__main__":
    config = {
                "img_channels": 3,
                "img_height": 114,
                "img_width": 10,
                "residual_num": 1,
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1.5e-6,
                "epoch": 50,
            }
    model = CsiNetAutoencoder(config, 16)

    # Create sample input with shape [batch_size, channels, height, width]
    inputs = torch.rand(size=(1, 3, 114, 10), dtype=torch.float32)  # Batch size of 1

    # Move model and input to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = inputs.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Measure inference time
    with torch.no_grad():  # Disable gradient calculation for inference
        if device.type == 'cuda':
            # Use CUDA events for GPU timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()  # Start timing
            output = model(inputs)  # Model inference
            end.record()  # End timing
            
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            # Calculate elapsed time
            inference_time_ms = start.elapsed_time(end)
            print(f"Inference Time in GPU: {inference_time_ms:.3f} ms")
        
        else:
            # Use time.time() for CPU timing
            start_time = time.time()
            output = model(inputs)  # Model inference
            end_time = time.time()
            
            # Calculate elapsed time in milliseconds
            inference_time_ms = (end_time - start_time) * 1000
            print(f"Inference Time: {inference_time_ms:.3f} ms")