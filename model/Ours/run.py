#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:34:06 2024

@author: jackson-devworks
"""
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import os

from .model import CloudSense, Discriminator
from .utils import nmse, calulate_error, compute_pck_pckh, NMSELoss



def main_CloudSense(data_loader, model_config, device, checkpoint_folder, check_compression_rate=False):
    torch.cuda.empty_cache()
    check_compression_rate = True
    os.makedirs(checkpoint_folder, exist_ok=True)

    model = CloudSense(model_config).to(device)
    dis = Discriminator().to(device)

    criterion_dis = nn.BCELoss().to(device)
    criterion_L2 = nn.MSELoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["lr"])
    optimizer_GAN = torch.optim.Adam(dis.parameters(), lr = 0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)
    pck_50_overall_max = 0
    torch.cuda.empty_cache()
    for epoch in tqdm(range(model_config['epoch'])):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        for idx, data in enumerate(data_loader['train']):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            optimizer_GAN.zero_grad()
            csi_data = data['input_wifi-csi']
            csi_data = csi_data.clone().detach().to(device)
            bs = csi_data.size(0)

    
            keypoint = data['output']
            xy_keypoint = keypoint[:,:,0:2].to(device)
            confidence = keypoint[:,:,2:3].to(device)
            
            vq_loss, z, reconstructed_csi, pred_xy_keypoint = model(csi_data)
            check_compression_rate = False
            #print(csi_data.size(), reconstructed_csi.size())
            ones_label=torch.ones(bs,1).to(device)
            output_real = dis(csi_data)[0]
            errD_real = criterion_dis(output_real, ones_label)
            
            zeros_label=torch.zeros(bs,1).to(device)
            output_fake = dis(reconstructed_csi)[0]
            errD_fake = criterion_dis(output_fake, zeros_label)
                
            hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
            loss = vq_loss + errD_fake + errD_real + hpe_loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer_GAN.step()
        avg_loss = total_loss / len(data_loader)
        model.adjust_codebook_based_on_loss(avg_loss, z)
        metric = []
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
                
                _, _, reconstructed_csi, pred_xy_keypoint = model(csi_data)
                #print(csi_data.size(), reconstructed_csi.size())
                #recontruction_loss = nmse(csi_data, reconstructed_csi)
                #hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
                #loss = model_config['lambda'] * recontruction_loss + hpe_loss
                
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
        if pck_50_overall > pck_50_overall_max:
            #print('saving the model at the end of epoch %d with pck_50: %.3f' % (epoch_index, pck_50_overall))
            torch.save(model, os.path.join(checkpoint_folder, "best.pt"))
            pck_50_overall_max = pck_50_overall
            print(f"\nBest Epoch in epoch {epoch} with {pck_50_overall_max}")
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
        
        _, _, reconstructed_csi, pred_xy_keypoint = model(csi_data)
        #print(csi_data.size(), reconstructed_csi.size())
        recontruction_loss = nmse(csi_data, reconstructed_csi)
        #hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
        #loss = _ + recontruction_loss + hpe_loss
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