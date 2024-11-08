#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:28:36 2024

@author: jackson-devworks
"""
import numpy as np
import torch
from torch import nn


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, original, recovered):
        loss = torch.mean(torch.square(original - recovered)) / torch.sum(
            torch.square(original)
        )
        return loss


def nmse(x, x_hat):
    return 10 * torch.log10(
        torch.mean(
            torch.mean(torch.square(x - x_hat), dim=(1, 2, 3))
            / torch.mean(torch.square(x), dim=(1, 2, 3))
        )
    )


def compute_pck_pckh(dt_kpts, gt_kpts, thr):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt = np.array(dt_kpts.detach().numpy())
    gt = np.array(gt_kpts.detach().numpy())
    assert dt.shape[0] == gt.shape[0]
    kpts_num = gt.shape[2]  # keypoints
    ped_num = gt.shape[0]  # batch_size
    # compute dist
    scale = np.sqrt(
        np.sum(np.square(gt[:, :, 5] - gt[:, :, 12]), 1)
    )  # right shoulder--left hip
    dist = (
        np.sqrt(np.sum(np.square(dt - gt), 1))
        / np.tile(scale, (gt.shape[2], 1)).T
    )
    # compute pck
    pck = np.zeros(gt.shape[2] + 1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100 * np.mean(dist[:, kpt_idx] <= thr)
        # compute average pck
    pck[17] = 100 * np.mean(dist <= thr)
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
    input_tensor: torch.Tensor, error_rate: float
) -> torch.Tensor:
    """
    Convert the input tensor to binary and simulate bit errors by flipping bits independently for each value.

    :param input_tensor: Input tensor containing integer values (shape: [batch_size, 56]).
    :param error_rate: The bit error rate (0 to 1), indicating the proportion of bits to flip.
    :return: The tensor with simulated bit errors, in binary representation.
    """
    batch_size, num_values = input_tensor.shape
    # Each value will be represented by 8 bits, so we prepare an 8-bit binary representation
    num_bits = 8

    # Convert each integer in the tensor to an 8-bit binary representation without using string formatting
    binary_tensor = torch.zeros(
        (batch_size, num_values * num_bits), dtype=torch.int32
    )
    for i in range(num_bits):
        binary_tensor[:, i::num_bits] = (input_tensor >> (7 - i)) & 1

    # Apply bit errors independently for each value
    noisy_tensor = binary_tensor.clone()
    total_bits = batch_size * num_values * num_bits
    num_bits_to_flip = int(total_bits * error_rate)  # Total bits to flip

    # Randomly select bit indices to flip
    indices_to_flip = torch.randint(0, total_bits, (num_bits_to_flip,))
    noisy_tensor.view(-1)[indices_to_flip] ^= 1  # Flip the bits

    # convert to decimal 
    noisy_tensor = binary_to_integer_tensor(noisy_tensor)
    decimal_tensor = binary_to_integer(noisy_tensor)
    
    return decimal_tensor


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

def binary_to_integer(binary_tensor):
    """
    Converts a binary tensor of shape [b, 56, 8] to an integer tensor of shape [b, 56].
    
    Args:
        binary_tensor (torch.Tensor): A tensor of shape [b, 56, 8], where each value is a binary bit (0 or 1).
        
    Returns:
        torch.Tensor: A tensor of shape [b, 56], where each element is the integer representation of the 8-bit binary number.
    """
    # Ensure the input is of the correct shape and type
    assert binary_tensor.shape[2] == 8, "The input tensor must have a size of 8 in the third dimension (bits)."
    assert binary_tensor.dtype == torch.int32, "The input tensor must be of type int32 (binary values)."
    
    # Convert binary tensor to integer by reshaping and applying bit shifting
    # Each bit is a power of 2, so we can multiply the bits by 2^position and sum them
    powers_of_two = torch.pow(2, torch.arange(7, -1, -1, dtype=torch.int32, device=binary_tensor.device))  # 2^7, 2^6, ..., 2^0
    integer_tensor = torch.matmul(binary_tensor, powers_of_two)  # This does the bit-wise sum
    
    return integer_tensor.to(torch.int64)

def binary_to_integer_tensor(binary_tensor):
    """
    Converts a binary tensor of shape [b, 448] into a tensor of shape [b, 56, 8], 
    where each 8 consecutive binary values represent one integer.
    
    Args:
        binary_tensor (torch.Tensor): A tensor of shape [b, 448], where each value is a binary bit (0 or 1).
        
    Returns:
        torch.Tensor: A tensor of shape [b, 56, 8], where each group of 8 bits represents a single integer.
    """
    # Ensure the input tensor is of shape [b, 448]
    assert binary_tensor.shape[1] == 448, "The input tensor must have a size of 448 in the second dimension."
    assert binary_tensor.dtype == torch.int32, "The input tensor must be of type int32 (binary values)."
    
    # Reshape the tensor to [b, 56, 8] (each group of 8 consecutive bits)
    reshaped_tensor = binary_tensor.view(binary_tensor.shape[0], 56, 8)
    
    return reshaped_tensor

if __name__ == "__main__":
    input_tensor = torch.randint(0, 256, (4, 56), dtype=torch.int64)  # Random binary tensor of shape [4, 448]
    decode_tensor = apply_bit_error(input_tensor, 0.9)
    print(input_tensor)
    print(decode_tensor)