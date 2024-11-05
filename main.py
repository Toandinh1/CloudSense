
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:02:33 2024

@author: jackson-devworks
"""
from sklearn.model_selection import train_test_split
import yaml
import torch
import os

from constant import experiment_config, skip_pipeline
from dataset_lib import make_dataset, make_dataloader


with open(experiment_config['mmfi_config'], 'r') as fd:  # change the .yaml file in your code.
    config = yaml.load(fd, Loader=yaml.FullLoader)
    
dataset_root = experiment_config['dataset_root']
train_dataset, test_dataset = make_dataset(dataset_root, config)
rng_generator = torch.manual_seed(config['init_rand_seed'])
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
val_data , test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)
val_loader = make_dataloader(val_data, is_training=False, generator=rng_generator, **config['val_loader'])
test_loader = make_dataloader(test_data, is_training=False, generator=rng_generator, **config['test_loader'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = {
    "train": train_loader,
    "valid": val_loader,
    "test" : test_loader    
}
for i in range(1):
    print("*" * 20)
    torch.cuda.empty_cache()
    print(f"Run {i+1}")
    for k, v in experiment_config['baseline'].items():
        if k in skip_pipeline:
            continue
        print("*"*10)
        print(f"Baseline {k}")
        pipeline = v["pipeline"]
        model_config = v['config']
        print(model_config)
        checkpoint_folder = os.path.join(experiment_config["checkpoint_folder"], k)
        os.makedirs(checkpoint_folder, exist_ok=True)
        pipeline(data_loader, model_config, device, checkpoint_folder)
        