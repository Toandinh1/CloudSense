#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:16:14 2024

@author: jackson-devworks
"""
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import os
from .model import LassoAutoEncoder
from .utils import nmse, calulate_error, compute_pck_pckh, NMSELoss, NMSE

theory_real_compressrate_dict = {
    "1": 1,
    #"34": 34,
    #"84": 84,
    #"1710": 1710
}

def main_LASSO(data_loader, model_config, device, all_checkpoint_folder):
    for k, v in theory_real_compressrate_dict.items():
        print("*"*5)
        print(f"with compress rate {k}: ")
        check_compression_rate = True
        checkpoint_folder = os.path.join(all_checkpoint_folder, k)
        os.makedirs(checkpoint_folder, exist_ok=True)
        model = LassoAutoEncoder(config=model_config, compression_rate=v).to(device)
        criterion_L2 = nn.MSELoss().to(device)
        ReconstructionLoss =  nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = model_config['lr'])
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
                
                reconstructed_csi, pred_xy_keypoint = model(csi_data, check_compression_rate)
                check_compression_rate = False
                #print(csi_data.size(), reconstructed_csi.size())
                recontruction_loss = ReconstructionLoss(csi_data, reconstructed_csi)
                hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
                loss = recontruction_loss + hpe_loss
                loss.backward()
                optimizer.step()
            
            nmse_min = 1e10
            pck_50_overall_max = 0
            metric = []
            pck_50_iter = []
            pck_40_iter = []
            pck_30_iter = []
            pck_20_iter = []
            pck_10_iter = []
            pck_5_iter = []
            avg_nmse = []
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
                    recontruction_loss = ReconstructionLoss(csi_data, reconstructed_csi)
                    hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
                    loss = recontruction_loss + hpe_loss
                    avg_nmse.append(recontruction_loss.item())

                    pred_xy_keypoint = pred_xy_keypoint.cpu()
                    xy_keypoint = xy_keypoint.cpu()
                    pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
                    xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
                     
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
                avg_nmse =  np.mean(avg_nmse)
                if pck_50_overall > pck_50_overall_max:
                   #print('saving the model at the end of epoch %d with pck_50: %.3f' % (epoch_index, pck_50_overall))
                   torch.save(model, os.path.join(checkpoint_folder, "best.pt"))
                   pck_50_overall_max = pck_50_overall
                torch.save(model, os.path.join(checkpoint_folder, "last.pt"))
            
            scheduler.step()
        
        model = torch.load(os.path.join(checkpoint_folder, "best.pt"), weights_only=False)
        metric = []
        avg_nmse = []
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
            loss = recontruction_loss + hpe_loss
            avg_nmse.append(recontruction_loss.item())
            pred_xy_keypoint = pred_xy_keypoint.cpu()
            xy_keypoint = xy_keypoint.cpu()
            pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
            xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
             
            metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
            pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck , 0.5))
            pck_40_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.4))
            pck_30_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.3))
            pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck , 0.2))
            pck_10_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.1))
            pck_5_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.05))  
        avg_nmse =  np.mean(avg_nmse)
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
        print('test result: nmse: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, pck_10: %.3f, pck_5: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f' % (avg_nmse, pck_50_overall,pck_40_overall, pck_30_overall,pck_20_overall, pck_10_overall,pck_5_overall, mpjpe_mean, pa_mpjpe_mean))