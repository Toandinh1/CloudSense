# CloudSense (TinySense)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Conference](https://img.shields.io/badge/IEEE%20PerCom-2026-red)](https://www.percom.org/)

**CloudSense** is the official implementation of **TinySense**, a cutting-edge compression framework designed for scalable and accurate Wi-Fi-based Human Pose Estimation (HPE). 

Accepted to **IEEE PerCom 2026**, this project addresses the communication bottleneck in cloud-based Wi-Fi sensing by partitioning the deep neural network (DNN) between local edge devices (e.g., Raspberry Pi) and cloud servers.

---

## 📝 Table of Contents

- [Abstract](#-abstract)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance](#-performance)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 📄 Abstract

Existing Wi-Fi sensing methods often transmit vast amounts of raw Channel State Information (CSI) data, straining network resources. **TinySense** introduces an efficient compression framework using a **Vector Quantization-based Generative Adversarial Network (VQGAN)**. By learning a compact discrete codebook, the data size is significantly reduced while maintaining the reconstruction quality required for reliable HPE.

Additionally, TinySense employs **K-means clustering** for dynamic bitrate adjustment and a **Transformer** model to predict missing indices, ensuring robustness even under unstable network conditions.

---

## ✨ Key Features

* **VQGAN-based Compression**: Compresses CSI data into compact latent representations using a learned codebook, achieving high compression rates with minimal accuracy loss.
* **Dual Selective Kernel (SK) Encoder**: Utilizes a lightweight Dual SK convolution module on the local device to effectively extract features from noisy Wi-Fi signals.
* **Adaptive Bitrate via K-Means**: Dynamically adjusts the codebook size using K-means clustering to balance compression quality and bandwidth availability.
* **Robust Error Recovery**: A second-stage **Transformer** predicts lost VQ indices during transmission, preventing reconstruction failures in unreliable networks.
* **Split Computing Architecture**: Offloads heavy computation (Decoder & Estimator) to the cloud while keeping the local Encoder lightweight.

---

## 🏗 System Architecture

The framework operates in a split-computing manner:

1.  **Local Device (Encoder):** Extracts latent features $Z$ from raw CSI $X$ using the Dual SK Encoder. These are mapped to discrete VQ indices $I$ via a codebook and binary-encoded for transmission.
2.  **Cloud Server (Decoder & Estimator):**
    * **Reconstruction:** The Decoder $G$ reconstructs the CSI data $\hat{X}$ from the received indices.
    * **Pose Estimation:** The Estimator $E_s$ predicts human pose keypoints from the quantized features.
    * **Error Correction:** A Transformer predicts any missing indices caused by packet loss.

---

## 📂 Directory Structure

```text
CloudSense/
├── Model/              # VQGAN, Encoder (Dual SK), Decoder, and Estimator definitions
├── SK_network/         # Selective Kernel (SK) attention module implementation
├── transformer/        # Transformer for lost VQ-indices prediction
├── trainer/            # Training scripts for VQGAN (Stage 1) and Transformer (Stage 2)
├── evaluation/         # Evaluation scripts (NMSE, PCK, MPJPE metrics)
├── configs/            # Configuration files (JSON/YAML) for bitrate and model settings
└── scripts/            # Utilities for data preprocessing (MM-Fi, WiPose)
