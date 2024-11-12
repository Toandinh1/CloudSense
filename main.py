#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:02:33 2024

@author: jackson-devworks
"""
import os

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constant import experiment_config, skip_pipeline
from dataset_lib import WiPoseDataset, make_dataloader, make_dataset
from model import run_tsne

if experiment_config["dataset_choice"] == 0:
    with open(
        experiment_config["data"][0]["mmfi_config"], "r"
    ) as fd:  # change the .yaml file in your code.
        config = yaml.load(fd, Loader=yaml.FullLoader)

    dataset_root = experiment_config["data"][0]["dataset_root"]
    train_dataset, test_dataset = make_dataset(dataset_root, config)
    rng_generator = torch.manual_seed(config["init_rand_seed"])
    train_loader = make_dataloader(
        train_dataset,
        is_training=True,
        generator=rng_generator,
        **config["train_loader"],
    )
    # testing_loader = make_dataloader(test_dataset, is_training=False, generator=rng_generator, **config['test_loader'])
    val_data, test_data = train_test_split(
        test_dataset, test_size=0.5, random_state=41
    )
    val_loader = make_dataloader(
        val_data,
        is_training=False,
        generator=rng_generator,
        **config["val_loader"],
    )
    test_loader = make_dataloader(
        test_data,
        is_training=False,
        generator=rng_generator,
        **config["test_loader"],
    )
    data_loader = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }
else:
    train_dataset = WiPoseDataset(experiment_config["data"][1]["dataset_root"])
    test_dataset = WiPoseDataset(
        experiment_config["data"][1]["dataset_root"], split="Test"
    )
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=41
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=experiment_config["data"][1]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=experiment_config["data"][1]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=experiment_config["data"][1]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    data_loader = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tnse_mode = True

if tnse_mode:
    run_tsne(
        data_loader,
        experiment_config["baseline"]["Our_GAN"]["config"],
        "/home/jackson-devworks/Desktop/CloudSense/output/MM-Fi/Our_GAN/9/best.pt",
        device,
        "./",
    )
    exit()


for i in range(1):
    print("*" * 20)
    torch.cuda.empty_cache()
    print(f"Run {i+1}")
    for k, v in experiment_config["baseline"].items():
        if k in skip_pipeline:
            continue
        print("*" * 10)
        print(f"Baseline {k}")
        pipeline = v["pipeline"]
        model_config = v["config"]
        checkpoint_folder = os.path.join(
            experiment_config["checkpoint_folder"],
            experiment_config["data"][experiment_config["dataset_choice"]][
                "name"
            ],
            k,
        )
        os.makedirs(checkpoint_folder, exist_ok=True)
        pipeline(data_loader, model_config, device, checkpoint_folder)
