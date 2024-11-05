#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:13:08 2024

@author: jackson-devworks
"""


import numpy as np 
import scipy.io
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
import torch.nn.functional as F

from .module import SKConv, SKUnit
from .utils import apply_bit_error, apply_bit_loss

class Quantizer(nn.Module):
    def __init__(self, latent_dim, num_embeddings):
        super(Quantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = latent_dim
        self.embedding = nn.Parameter(torch.randn(num_embeddings, latent_dim))

    def forward(self, z):
        z_flat = z.view(-1, self.embedding_dim)
        distances = (z_flat.pow(2).sum(1, keepdim=True) 
                     - 2 * z_flat @ self.embedding.t() 
                     + self.embedding.pow(2).sum(1))
        indices = distances.argmin(1)
        z_quantized = self.embedding[indices].view(*z.shape)
        return z_quantized, indices

    def update_codebook(self, z, new_num_embeddings):
        if new_num_embeddings != self.num_embeddings:
            self.num_embeddings = new_num_embeddings
            self.embedding = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim).to(self.embedding.device))
        
        z_flat = z.view(-1, self.embedding_dim).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_embeddings, random_state=0).fit(z_flat)
        new_embeddings = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.embedding.data = new_embeddings.to(self.embedding.device)
        print(f"\nCodebook updated: {self.num_embeddings} vectors in codebook")
    def decode(self, indices, shape):
        """
        Decode from indices to reconstruct the quantized tensor.
        
        Parameters:
        indices (torch.Tensor): The indices tensor from the quantizer.
        shape (tuple): The shape of the original input tensor.

        Returns:
        torch.Tensor: The reconstructed tensor from the codebook indices.
        """
        # Use the codebook embeddings to map indices back to the quantized vectors
        z_reconstructed = self.embedding[indices].view(shape)
        return z_reconstructed
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.SKunit1 = SKUnit(in_features=3, mid_features=16, out_features=32, dim1 = 114,dim2 = 10,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.SKunit2 = SKUnit(in_features=32, mid_features=64, out_features=128, dim1 = 57,dim2 = 8,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.SKunit3 = SKUnit(in_features=128, mid_features=64, out_features=32, dim1 = 28,dim2 = 2,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.pool_1 = nn.AvgPool2d(2)
        self.pool_2 = nn.AvgPool2d(2)
        # output size: (128,96,6,6)
    def forward(self,x):
        encoder_1_output = self.SKunit1(x)
        pool_1_output = self.pool_1(encoder_1_output)
        encoder_2_output = self.SKunit2(pool_1_output)
        encoder_output = self.pool_2(encoder_2_output)
        encoder_output = self.SKunit3(encoder_output)

        return encoder_output
    
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.SKunit1 = SKUnit(in_features=3, mid_features=16, out_features=32, dim1 = 114,dim2 = 10,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
    self.SKunit2 = SKUnit(in_features=32, mid_features=64, out_features=128, dim1 = 57,dim2 = 8,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
    self.SKunit3 = SKUnit(in_features=128, mid_features=64, out_features=32, dim1 = 28,dim2 = 2,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
    self.pool_1 = nn.AvgPool2d(2)
    self.pool_2 = nn.AvgPool2d(2)
    
    self.fc1=nn.Linear(32*28*2,512)
    self.bn=nn.BatchNorm1d(512,momentum=0.9)
    self.fc2=nn.Linear(512,1)
    self.sigmoid=nn.Sigmoid()
    self.relu = nn.LeakyReLU(0.2)

  def forward(self,x):
    batch = x.shape[0]
    encoder_1_output = self.SKunit1(x)
    pool_1_output = self.pool_1(encoder_1_output)
    encoder_2_output = self.SKunit2(pool_1_output)
    encoder_output = self.pool_2(encoder_2_output)
    encoder_output = self.SKunit3(encoder_output)
    x=encoder_output.view(batch,-1)
    x1=x;
    x=self.relu(self.bn(self.fc1(x)))
    x=self.sigmoid(self.fc2(x))

    return x,x1    

class regression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(regression, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.transposed_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.transposed_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.transposed_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(3)

        self.pool = nn.AdaptiveAvgPool2d((114,10))
        #self.transposed_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.transposed_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.transposed_conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.transposed_conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        return x
    
class CloudSense(nn.Module):
    def __init__(self, config):
        super(CloudSense, self).__init__()
        embedding_dim = config["embedding_dim"]
        commitment_cost = config["commitment_cost"]
        self.min_codebook_size = config["min_codebook_size"]
        self.max_codebook_size = config["max_codebook_size"]
        self.change_step = config["change_step"]
        
        self.unreliable_mode = config["unreliable_mode"]
        self.unrilable_rate_in_training = config["unrilable_rate_in_training"]
        
        self.prev_loss = None

        self._encoder = Encoder()
        self._pre_vq_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=embedding_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )
        
        self._trans_vq_vae = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embedding_dim, out_channels=96, kernel_size=1, stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )  
        
        self.regression = regression(input_dim=embedding_dim * 28 * 2, output_dim=34, hidden_dim=32)
        self._decoder = Decoder()
        self.vq = Quantizer(latent_dim=embedding_dim, num_embeddings=256)
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

    def adjust_codebook_based_on_loss(self, current_loss, z):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return

        if current_loss < self.prev_loss:  # If loss improved
            new_codebook_size = max(self.min_codebook_size, self.vq.num_embeddings - self.change_step)
        else:  # If loss did not improve
            new_codebook_size = min(self.max_codebook_size, self.vq.num_embeddings + self.change_step)

        if new_codebook_size != self.vq.num_embeddings:
            self.vq.update_codebook(z, new_codebook_size)
        
        self.prev_loss = current_loss
            
    def forward(self, x):
        batch = x.shape[0]
        
        # Encode input
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        
        # Quantization
        z_quantized, indices = self.vq(z)
        
        if self.unreliable_mode == 0:
            noisy_indices = apply_bit_error(indices.cpu(), error_rate=self.unrilable_rate_in_training)
        elif self.unreliable_mode == 1: 
            noisy_indices = apply_bit_loss(indices.cpu(), loss_rate=self.unrilable_rate_in_training)
        else:
            noisy_indices = indices
            
        # Lưu toàn bộ tensor vào file văn bản
        z_reconstructed = self.vq.decode(noisy_indices, z.shape)
        
        #print(F.mse_loss(z_quantized, z_reconstructed), noisy_indices.shape)
        # Reshape quantized vector for decoder and regression
        z_quantized = z_reconstructed.view(batch, self.embedding_dim, 28, 2)

        # Calculate reconstructed output and regression prediction
        y_p = self.regression(z_quantized)
        r_x = self._decoder(z_quantized)
        
        # Reshape regression output
        y_p = y_p.reshape(batch, 17, 2)
        
        # Compute losses
        reconstruction_loss = F.mse_loss(r_x, x)
        
        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z, z_quantized.detach())
        
        # Codebook loss (encourage embedding vectors to be close to quantized ones)
        codebook_loss = F.mse_loss(z.detach(), z_quantized)
        
        # Total loss
        vq_loss = reconstruction_loss + commitment_loss + codebook_loss
        
        return vq_loss, z, r_x, y_p
