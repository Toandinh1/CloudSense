#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:16:35 2024

@author: jackson-devworks
"""

from model import main_RSCNet, main_EfficientFi, main_CSINet

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
            }
        },
        "EfficientFi": {
            "pipeline": main_EfficientFi,
            "config": {
                "embedding_dim": 256,
                "commitment_cost": 1,
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1e-5,
                "epoch": 20,
            }
        }, 
        "CSINet": {
            "pipeline": main_CSINet,
            "config": {
                "img_channels":3,
                "img_height": 114,
                "img_width": 10,
                "residual_num": 1,
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 1.5e-6,
                "epoch": 20,
            }
        }
    }
}

skip_pipeline = []