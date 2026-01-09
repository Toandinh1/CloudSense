#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:18:23 2024

@author: jackson-devworks
"""
import numpy as np 
from sklearn.linear_model import Lasso
import torch
from torch import nn

def lasso_compress(csi_tensor, alpha=0.1):
    batch_size = csi_tensor.size(0)
    csi_np = csi_tensor.view(batch_size, -1).cpu().numpy()  # Flatten for LASSO

    lasso = Lasso(alpha=0.1)  # Bạn có thể điều chỉnh alpha để thay đổi mức độ nén

    # Huấn luyện mô hình
    lasso.fit(csi_np, csi_np)
    
    # Dự đoán (nén) dữ liệu
    compressed_data = lasso.predict(csi_np)

    compressed_tensor = torch.tensor(compressed_data, dtype=torch.float32).view(batch_size, -1)
    return compressed_tensor.to(csi_tensor.device)


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