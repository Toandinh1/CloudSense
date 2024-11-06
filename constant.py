#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:16:35 2024

@author: jackson-devworks
"""

from model import (
    main_CloudSense,
    main_CSINet,
    main_DeepCMC,
    main_EfficientFi,
    main_LASSO,
    main_RSCNet,
)

experiment_config = {
    "mmfi_config": "/home/jackson-devworks/Desktop/CloudSense/dataset_lib/config.yaml",
    "dataset_root": "/home/jackson-devworks/Desktop/HPE/Dataset",
    "checkpoint_folder": "/home/jackson-devworks/Desktop/CloudSense/output",
    "baseline": {
        "RSCNet": {
            "pipeline": main_RSCNet,
            "config": {
                "recurrent_block": 256,
                "expansion": 1,
                "num_frame": 57,
                "lambda": 50,
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1.5e-6,
                "epoch": 20,
            },
        },
        "EfficientFi": {
            "pipeline": main_EfficientFi,
            "config": {
                "embedding_dim": 32,
                "commitment_cost": 1,
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1e-5,
                "epoch": 20,
            },
        },
        "CSINet": {
            "pipeline": main_CSINet,
            "config": {
                "img_channels": 3,
                "img_height": 114,
                "img_width": 10,
                "residual_num": 1,
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1.5e-6,
                "epoch": 20,
            },
        },
        "LASSO": {
            "pipeline": main_LASSO,
            "config": {
                "input_shape": (3, 114, 10),
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1.5e-6,
                "epoch": 30,
            },
        },
        "DeepCMC": {
            "pipeline": main_DeepCMC,
            "config": {
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1.5e-6,
                "epoch": 20,
            },
        },  
        "Ours": {
            "pipeline": main_CloudSense,
            "config": {
                "min_codebook_size": 64,
                "max_codebook_size": 1000,
                "initial_cook_size": 999,
                "change_step": 100,
                "embedding_dim": 256,
                "commitment_cost": 1,
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1e-5,
                "epoch": 10,
                "unreliable_mode": 0,  # 0: bit error   #1: bit loss
                "unrilable_rate_in_training": 0.1,
            },
        },
    },
}

skip_pipeline = ["LASSO", "RSCNet", "CSINet", "DeepCMC", "EfficientFi"]
