#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:26:54 2024

@author: jackson-devworks
"""
from torch import nn
import torch
from .module import Encoder, Quantize, Decoder, HumanPoseEstimator


def compute_compression_rate(original_tensor: torch.Tensor, compressed_tensor: torch.Tensor):
    """
    Compute the compression rate of a tensor.

    Parameters:
    - original_tensor: The original tensor before compression.
    - compressed_tensor: The tensor after compression.

    Returns:
    - A tensor representing the compression rate.
    """
    print(original_tensor.size(), compressed_tensor.size())
    # Get the sizes of the original and compressed tensors
    original_size = original_tensor.numel()  # Total number of elements in the original tensor
    compressed_size = compressed_tensor.numel()  # Total number of elements in the compressed tensor

    # Compute compression rate
    compression_rate = original_size / compressed_size if compressed_size > 0 else torch.tensor(float('inf'))  # Handle division by zero

    return compression_rate

compress_rate_fc_dim_dict = {
    1/0.75: 3200,
    1/21: 100
    
}
class EfficientFi(nn.Module):
    def __init__(self, config, check_compression_rate=False, compression_rate=1/36.0):
        super(EfficientFi, self).__init__()
        self.check_compression_rate = check_compression_rate
        self.num_embeddings = config["num_embeddings"]
        self.embedding_dim = config["embedding_dim"]
        self.commitment_cost = config["commitment_cost"]
        
        self._encoder = Encoder(compression_rate)
        self._pre_vq_conv = nn.Conv2d(in_channels=int(96 * compression_rate), out_channels=self.embedding_dim, kernel_size=1, stride=1)
        self._vq_vae = Quantize(self.num_embeddings, self.embedding_dim, self.commitment_cost)
        self._trans_vq_vae = nn.ConvTranspose2d(in_channels=self.embedding_dim, out_channels=int(96 * compression_rate), kernel_size=1, stride=1)
        self._decoder = Decoder(compression_rate)
        self.hpe_estimator = HumanPoseEstimator(compress_rate_fc_dim_dict[compression_rate], 34, 32)
    
    def forward(self, x):
        batch_size = x.shape[0]
        z, indices1, indices2 = self._encoder(x)
        vq = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(vq)
        latent = self._trans_vq_vae(quantized)
        
        if self.check_compression_rate:
            print(f"Compression rate: {compute_compression_rate(x, latent)}")
        
        r_x = self._decoder(latent, indices2, indices1)
        pred_keypoint = self.hpe_estimator(latent).reshape(batch_size, 17, 2)

        return loss, r_x, pred_keypoint