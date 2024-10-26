#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:37:23 2024

@author: jackson-devworks
"""
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import os

from .model import RSCNet
from .utils import nmse, compute_pck_pckh, calulate_error

def main_RSCNet(data_loader, model_config, device, checkpoint_folder):
    model = RSCNet(config=model_config).to(device)
    criterion_L2 = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=model_config["lr"], 
                            momentum=model_config["momentum"], 
                            weight_decay=model_config["weight_decay"]
                            )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, model_config['epoch'])
    torch.cuda.empty_cache()
    for epoch in tqdm(range(model_config['epoch'])):
        torch.cuda.empty_cache()
        model.train()
        for idx, data in enumerate(data_loader['train']):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            csi_data = data['input_wifi-csi']
            csi_data = csi_data.clone().detach().to(device)

    
            keypoint = data['output']
            xy_keypoint = keypoint[:,:,0:2].to(device)
            confidence = keypoint[:,:,2:3].to(device)
            
            reconstructed_csi, pred_xy_keypoint = model(csi_data)
            #print(csi_data.size(), reconstructed_csi.size())
            recontruction_loss = nmse(csi_data, reconstructed_csi)
            hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
            loss = model_config['lambda'] * recontruction_loss + hpe_loss
            loss.backward()
            optimizer.step()
        
        valid_loss_iter = 0
        pck_50_overall_max = 0
        test_loss_iter = []
        metric = []
        time_iter = []
        pck_50_iter = []
        pck_40_iter = []
        pck_30_iter = []
        pck_20_iter = []
        pck_10_iter = []
        pck_5_iter = []
        with torch.no_grad():
            torch.cuda.empty_cache()
            model.eval()
            for idx, data in enumerate(data_loader['valid']):
                torch.cuda.empty_cache()
                csi_data = data['input_wifi-csi']
                csi_data = csi_data.clone().detach().to(device)

        
                keypoint = data['output']
                xy_keypoint = keypoint[:,:,0:2].to(device)
                confidence = keypoint[:,:,2:3].to(device)
                
                reconstructed_csi, pred_xy_keypoint = model(csi_data)
                #print(csi_data.size(), reconstructed_csi.size())
                recontruction_loss = nmse(csi_data, reconstructed_csi)
                hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
                loss = model_config['lambda'] * recontruction_loss + hpe_loss
                
                pred_xy_keypoint = pred_xy_keypoint.cpu()
                xy_keypoint = xy_keypoint.cpu()
                pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
                xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
                pck = compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
                 
                metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
                pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5))
                pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2))
            mean = np.mean(metric, 0)*1000
            mpjpe_mean = mean[0]
            pa_mpjpe_mean = mean[1]
            pck_50 = np.mean(pck_50_iter,0)
            pck_20 = np.mean(pck_20_iter,0)
            pck_50_overall = pck_50[17]
            pck_20_overall = pck_20[17]
            if pck_50_overall > pck_50_overall_max:
               #print('saving the model at the end of epoch %d with pck_50: %.3f' % (epoch_index, pck_50_overall))
               torch.save(model, os.path.join(checkpoint_folder, "best.pt"))
               pck_50_overall_max = pck_50_overall
        scheduler.step()
    
    model = torch.load(os.path.join(checkpoint_folder, "best.pt"), weights_only=False)
    metric = []
    torch.cuda.empty_cache()
    model.eval()
    for idx, data in enumerate(data_loader['test']):
        torch.cuda.empty_cache()
        csi_data = data['input_wifi-csi']
        csi_data = csi_data.clone().detach().to(device)


        keypoint = data['output']
        xy_keypoint = keypoint[:,:,0:2].to(device)
        confidence = keypoint[:,:,2:3].to(device)
        
        reconstructed_csi, pred_xy_keypoint = model(csi_data)
        #print(csi_data.size(), reconstructed_csi.size())
        recontruction_loss = nmse(csi_data, reconstructed_csi)
        hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
        loss = model_config['lambda'] * recontruction_loss + hpe_loss
        
        pred_xy_keypoint = pred_xy_keypoint.cpu()
        xy_keypoint = xy_keypoint.cpu()
        pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
        xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
        pck = compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
         
        metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
        pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck , 0.5))
        pck_40_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.4))
        pck_30_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.3))
        pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck , 0.2))
        pck_10_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.1))
        pck_5_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.05))  
            
    mean = np.mean(metric, 0)*1000
    mpjpe_mean = mean[0]
    pa_mpjpe_mean = mean[1]
    pck_50 = np.mean(pck_50_iter, 0)
    pck_40 = np.mean(pck_40_iter, 0)
    pck_30 = np.mean(pck_30_iter, 0)
    pck_20 = np.mean(pck_20_iter, 0)
    pck_10 = np.mean(pck_10_iter, 0)
    pck_5 = np.mean(pck_5_iter, 0)
    pck_50_overall = pck_50[17]
    pck_40_overall = pck_40[17]
    pck_30_overall = pck_30[17]
    pck_20_overall = pck_20[17]
    pck_10_overall = pck_10[17]
    pck_5_overall = pck_5[17]
    print('test result: pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, pck_10: %.3f, pck_5: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f' % (pck_50_overall,pck_40_overall, pck_30_overall,pck_20_overall, pck_10_overall,pck_5_overall, mpjpe_mean, pa_mpjpe_mean))