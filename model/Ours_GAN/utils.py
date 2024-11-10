#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:28:36 2024

@author: jackson-devworks
"""
import numpy as np
import torch
from torch import nn


def NMSELoss(input, target):
    
    # Calculate squared error
    squared_error = (input - target) ** 2
        
    # Calculate mean squared error
    mse = torch.mean(squared_error)

    # Calculate normalized mean squared error
    # Divide mean squared error by the variance of the target
    # This helps in normalizing the loss across different scales of the target
    nmse = mse / torch.var(target)

    return nmse


def nmse(x, x_hat):
    return 10 * torch.log10(
        torch.mean(
            torch.mean(torch.square(x - x_hat), dim=(1, 2, 3))
            / torch.mean(torch.square(x), dim=(1, 2, 3))
        )
    )


def compute_pck_pckh(dt_kpts,gt_kpts,thr):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt=np.array(dt_kpts)
    gt=np.array(gt_kpts)
    assert(dt.shape[0]==gt.shape[0])
    kpts_num=gt.shape[2] #keypoints
    ped_num=gt.shape[0] #batch_size
    #compute dist
    scale=np.sqrt(np.sum(np.square(gt[:,:,1]-gt[:,:,11]),1)) #right shoulder--left hip
    #dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    #dist=np.sqrt(np.sum(np.square(dt-gt),1))
    #compute pck
    pck = np.zeros(gt.shape[2]+1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100*np.mean(dist[:,kpt_idx] <= thr)
        # compute average pck
    pck[17] = 100*np.mean(dist <= thr)
    return pck


def compute_pck_pckh_18(dt_kpts,gt_kpts,thr):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt=np.array(dt_kpts)
    gt=np.array(gt_kpts)
    assert(dt.shape[0]==gt.shape[0])
    kpts_num=gt.shape[2] #keypoints
    ped_num=gt.shape[0] #batch_size
    #compute dist
    scale=np.sqrt(np.sum(np.square(gt[:,:,5]-gt[:,:,8]),1)) #right shoulder--left hip
    dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    #dist=np.sqrt(np.sum(np.square(dt-gt),1))
    #compute pck
    pck = np.zeros(gt.shape[2]+1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100*np.mean(dist[:,kpt_idx] <= thr)
        # compute average pck
    pck[18] = 100*np.mean(dist <= thr)
    return pck


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Args:
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.0).sum()
    ssY = (Y0**2.0).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    X0 /= normX
    Y0 /= normY

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    detT = np.linalg.det(T)
    if detT < 0:  # Ensure a proper rotation matrix
        V[:, -1] *= -1
        s[-1] *= -1
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def calulate_error(predicted_keypoints, ground_truth_keypoints):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.

    :param predicted_keypoints: Estimated results from the algorithm, shape=[n, h, w]
    :param ground_truth_keypoints: Ground truth marked results, shape=[n, h, w]
    :return: Mean Per Joint Position Error (MPJPE) and Procrustes Aligned MPJPE (PA-MPJPE)
    """
    # Convert inputs to numpy arrays
    predicted_keypoints = np.array(predicted_keypoints.detach().numpy())
    ground_truth_keypoints = np.array(ground_truth_keypoints.detach().numpy())

    # Validate input shapes
    assert (
        predicted_keypoints.shape == ground_truth_keypoints.shape
    ), "Input shapes must match"

    N = predicted_keypoints.shape[0]  # Number of samples
    num_joints = predicted_keypoints.shape[2]  # Number of keypoints

    # Calculate MPJPE
    mpjpe = np.mean(
        np.sqrt(
            np.sum(
                np.square(predicted_keypoints - ground_truth_keypoints), axis=2
            )
        )
    )

    # Calculate PA-MPJPE
    pampjpe = np.zeros(N)

    for n in range(N):
        frame_pred = predicted_keypoints[n]  # Shape [h, w]
        frame_gt = ground_truth_keypoints[n]  # Shape [h, w]

        # Compute similarity transform
        _, Z, T, b, c = compute_similarity_transform(
            frame_gt, frame_pred, compute_optimal_scale=True
        )

        # Apply the transformation to predictions
        frame_pred_transformed = (b * frame_pred @ T) + c
        pampjpe[n] = np.mean(
            np.sqrt(
                np.sum(np.square(frame_pred_transformed - frame_gt), axis=1)
            )
        )

    # Compute average PA-MPJPE
    pampjpe = np.mean(pampjpe)

    return mpjpe, pampjpe


def apply_bit_error(
    input_tensor: torch.Tensor, error_rate: float, num_embedding
) -> torch.Tensor:
    keep_rate = 1 - error_rate
    mask = torch.bernoulli(keep_rate * torch.ones(input_tensor.shape, device=input_tensor.device))
    mask = mask.round().to(dtype=torch.int64)
    random_indices = torch.randint_like(input_tensor, num_embedding)
    new_indices = mask * input_tensor + (1 - mask) * random_indices
    return new_indices

def apply_bit_loss(tensor: torch.Tensor, loss_rate: float) -> torch.Tensor:
    """
    Mô phỏng mất bit bằng cách loại bỏ ngẫu nhiên các bit trong tensor.

    :param tensor: Tensor đầu vào có kích thước 14336 kiểu torch.Tensor (int64).
    :param loss_rate: Tỉ lệ mất bit (0 đến 1).
    :return: Tensor với mất bit đã được mô phỏng.
    """
    # assert tensor.size(0) == 14336, "Tensor phải có kích thước 14336."

    # Xác định số lượng bit cần loại bỏ
    num_bits_to_remove = int(loss_rate * tensor.size(0))

    # Chọn ngẫu nhiên các chỉ số để loại bỏ
    indices_to_remove = torch.randint(0, tensor.size(0), (num_bits_to_remove,))

    # Sử dụng `set` để loại bỏ các chỉ số
    indices_to_keep = torch.tensor(
        [
            i
            for i in range(tensor.size(0))
            if i not in indices_to_remove.tolist()
        ]
    )

    return tensor[indices_to_keep]


