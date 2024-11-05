#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:28:23 2024

@author: jackson-devworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantize(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, commitment_cost):
        super(Quantize, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding  = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        
    def forward(self, inputs): 
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)#16384x1 
                    + torch.sum(self._embedding.weight**2, dim=1)#1 x _num_embeddings
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        # 16384 x _num_embeddings
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs) 
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.encoder_1 = nn.Sequential(
            #input size: (3,114,500)
            nn.Conv2d(3,32,(15,23),stride=9),
            nn.ReLU(inplace=False),
            nn.Conv2d(32,32,(3,7),stride=1),
            nn.ReLU(inplace=False)
            #output size: (32,10,48)
        )
        self.pool_1 = nn.MaxPool2d((1,2),stride=(1,2),return_indices=True)
        # output size: (32,10,24)
        self.encoder_2 = nn.Sequential(
            #input size: (32,10,24)
            nn.Conv2d(32,64,(3,7),stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64,96,(3,7),stride=1),
            nn.ReLU(inplace=False)
            #output size: (128,96,6,12)
        )
        self.pool_2 = nn.MaxPool2d((1,2),stride=(1,2),return_indices=True)
        # output size: (128,96,6,6)
    def forward(self,x):
        encoder_1_output = self.encoder_1(x)
        pool_1_output,indices1 = self.pool_1(encoder_1_output)
        encoder_2_output = self.encoder_2(pool_1_output)
        encoder_output,indices2 = self.pool_2(encoder_2_output)
        return encoder_output, indices1, indices2

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.unpool_2 = nn.MaxUnpool2d((1,2),stride=(1,2))
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(96,64,(3,7),stride=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64,32,(3,7),stride=1)
        )
        self.unpool_1 = nn.MaxUnpool2d((1,2),stride=(1,2))
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(32,32,(3,7),stride=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32,3,(15,23),stride=9)
        )
    def forward(self, x, indices2, indices1):
        r_x = self.unpool_2(x,indices2)
        r_x = self.decoder_2(r_x)
        r_x = self.unpool_1(r_x,indices1)
        r_x = self.decoder_1(r_x)
        return r_x

class HumanPoseEstimator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(HumanPoseEstimator, self).__init__()
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