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
    def __init__(self, output_hpe = 36):
        super(DeepCMC, self).__init__()
        self.encoder = FeatureEncoder()
        self.decoder = FeatureDecoder()
        self.human_pose_estimator = HumanPoseEstimator(1024, output_hpe, 32)

    def forward(self, x, check_compression_rate = False):
        batch = x.shape[0]
        quantized, indices = self.encoder(x)
        if check_compression_rate:
            with torch.no_grad():
                compression_rate = compute_compression_rate(x, quantized)
            
            print(f"\nEncoder size: {quantized.size()}")
            print(f"Compression rate: {compression_rate}")
        decoded = self.decoder(quantized)
        red_keypoint = self.human_pose_estimator(quantized).reshape(batch, -1, 2)
        r_x = F.interpolate(decoded, size=(114,10), mode="bilinear", align_corners=True)
        return decoded, red_keypoint
    

if __name__ == "__main__":
    model = DeepCMC()

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

    # print(output)