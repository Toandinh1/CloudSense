#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:26:54 2024

@author: jackson-devworks
"""
import torch
import torch.nn.functional as F
from torch import nn
from vector_quantize_pytorch import ResidualVQ, VectorQuantize

from .module import Decoder, Encoder, HumanPoseEstimator, Quantize
from .utils import apply_bit_error, apply_bit_loss


def compute_compression_rate(
    original_tensor: torch.Tensor, compressed_tensor: torch.Tensor
):
    """
    Compute the compression rate of a tensor.

    Parameters:
    - original_tensor: The original tensor before compression.
    - compressed_tensor: The tensor after compression.

    Returns:
    - A tensor representing the compression rate.
    """
    # print(original_tensor.size(), compressed_tensor.size())
    # Get the sizes of the original and compressed tensors
    original_size = (
        original_tensor.numel()
    )  # Total number of elements in the original tensor
    compressed_size = (
        compressed_tensor.numel()
    )  # Total number of elements in the compressed tensor

    # Compute compression rate
    compression_rate = (
        original_size / compressed_size
        if compressed_size > 0
        else torch.tensor(float("inf"))
    )  # Handle division by zero

    return compression_rate


class EfficientFi(nn.Module):
    def __init__(self, config, compress_rate, output_hpe = 36):
        super(EfficientFi, self).__init__()
        self.num_embeddings = config["num_embeddings"]
        self.embedding_dim = config["embedding_dim"]
        self.commitment_cost = config["commitment_cost"]
        self.unreliable_mode = config["unreliable_mode"]
        self._encoder = Encoder()
        self._pre_vq_conv = nn.Conv2d(
            in_channels=96,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
        )
        self._vq_vae = VectorQuantize(
            dim=25,
            codebook_size=self.num_embeddings,
            commitment_weight=self.commitment_cost,
        )
        self._trans_vq_vae = nn.ConvTranspose2d(
            in_channels=self.embedding_dim,
            out_channels=96,
            kernel_size=1,
            stride=1,
        )
        self._decoder = Decoder()
        self.hpe_estimator = HumanPoseEstimator(2400, output_hpe, 32)

    def forward(self, x, is_test=False, error_rate=None):
        batch_size = x.shape[0]
        z, indices1, indices2 = self._encoder(x)
        vq = self._pre_vq_conv(z)
        # print(vq.shape)

        vq = vq.view(batch_size, -1,25)
        z_quantized, indices, vq_loss = self._vq_vae(vq)
        if is_test:
            if self.unreliable_mode == 0:
                noisy_indices = apply_bit_error(
                    indices.cpu(),
                    error_rate=error_rate,
                    num_embedding=self.vq.codebook_size,
                )
                # print(f"Noise index: {indices}")
                # print(f"noisy index: {noisy_indices}")
            elif self.unreliable_mode == 1:
                noisy_indices = apply_bit_loss(
                    indices.cpu(), loss_rate=error_rate
                )
            else:
                noisy_indices = indices
        else:
            if self.unreliable_mode == 0:
                noisy_indices = apply_bit_error(
                    indices.cpu(),
                    error_rate=self.unrilable_rate_in_training,
                    num_embedding=self.vq.codebook_size,
                )
            elif self.unreliable_mode == 1:
                noisy_indices = apply_bit_loss(indices.cpu(), loss_rate=0)
            else:
                noisy_indices = indices
        noisy_indices = noisy_indices.cuda()
        z_reconstructed = self._vq_vae.get_codes_from_indices(noisy_indices)
        z_reconstructed = z_reconstructed.view(batch_size, -1, 5, 5)
        # print(z_reconstructed.shape)
        latent = self._trans_vq_vae(z_reconstructed)

        r_x = self._decoder(latent, indices2, indices1)
        # r_x = F.interpolate(
        #     r_x, size=(114, 10), mode="bilinear", align_corners=True
        # )
        pred_keypoint = self.hpe_estimator(latent).reshape(batch_size, -1, 2)

        return vq_loss, r_x, pred_keypoint
