#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:14:27 2024

@author: jackson-devworks
"""

from torch import nn
import torch
import numpy as np

from .module import Encoder, RecurrentBlock, Decoder, HumanPoseEstimator


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

class RSCNet(nn.Module):
    def __init__(self, config=True, compression_rate = 1, output_hpe = 36):
        super(RSCNet, self).__init__()
        self.compression_rate = compression_rate
        self.config = config
        self.num_frames = config["num_frame"]
        self.input_shape = [3, self.num_frames, 10]
        self.sequence_length = 125//self.num_frames
        # Encoder in AP
        self.encoder = Encoder(input_shape=self.input_shape)
        self.input_size = np.prod(self.input_shape)
        self.encoder_fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1710 , int(self.input_size/self.compression_rate)),
		)
        
        # Recurrent block before passing through decoder - reconstruction or downstreamtask - HPE
        c,w,h = self.input_shape
        self.embedding_size = c*w*h//self.compression_rate
        self.recurrent_block = RecurrentBlock(self.embedding_size, self.config["recurrent_block"])
        
        # Decoder in Cloud
        self.decoder_fc = nn.Sequential(
			nn.Linear(self.config["recurrent_block"], int(self.input_size)),
			nn.Unflatten(1, (self.input_shape)),
		)
        self.decoder = Decoder(self.input_shape, self.config['expansion'])

        # HPE Task
        self.human_pose_estimator = HumanPoseEstimator(512, output_hpe, 32)
        
    def forward(self, x, check_compression_rate = False):
        batch_size = x.shape[0]
        #print(f"x shape: {x.size()}")
        new_x = x.permute(0,2,1,3).contiguous()
		# batch x 250 x 1 x 90
        new_x = new_x.view(batch_size*self.sequence_length,self.num_frames,3,10)
		# (batch x t) x num_frames x 1 x 90
        new_x=new_x.permute(0,2,1,3)
        # Encode status
        z_e = self.encoder(new_x)
        c = self.encoder_fc(z_e)
        seq_c = c.view(batch_size, self.sequence_length, -1)

        if check_compression_rate:
            print(f"\nEncoder size: {seq_c.size()}")
            print(f"Compression rate: {compute_compression_rate(x, seq_c)}")
        # Recurrent block
        seq_c_r_d, _ = self.recurrent_block(seq_c)
        seq_c_r_d = seq_c_r_d.contiguous()
        c_r_d = seq_c_r_d.view(batch_size*self.sequence_length, -1)
        
        #print(f"c_r_d size: {c_r_d.size()}")
        # Decoder status
        z_d = self.decoder_fc(c_r_d)
        x_hat = self.decoder(z_d)
        new_x_hat = x_hat.permute(0,2,1,3).contiguous()
        new_x_hat = new_x_hat.view(batch_size, 114, 3, 10)
        new_x_hat =new_x_hat.permute(0,2,1,3).contiguous()
        #print(f"Reconstructed data size: {new_x_hat.size()}")
        
        # HPE task
        pred_keypoint = self.human_pose_estimator(c_r_d.view(batch_size, -1))
        pred_keypoint = pred_keypoint.reshape(batch_size,-1,2)

        return new_x_hat, pred_keypoint
        