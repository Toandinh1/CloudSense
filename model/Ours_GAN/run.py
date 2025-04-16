#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:34:06 2024

@author: jackson-devworks
"""
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .model import CloudSense, Discriminator, TSNECloudSense
from .utils import NMSELoss, calulate_error, compute_pck_pckh, nmse


def main_CloudSenseGAN(
    data_loader,
    model_config,
    device,
    all_checkpoint_folder,
    check_compression_rate=False,
):
    torch.cuda.empty_cache()
    for code_book_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        torch.cuda.empty_cache()
        checkpoint_folder = os.path.join(
            all_checkpoint_folder, "wipose", str(code_book_size)
        )
        os.makedirs(checkpoint_folder, exist_ok=True)
        model_config["min_codebook_size"] = code_book_size
        model_config["initial_cook_size"] = code_book_size
        print(model_config)
        model = CloudSense(model_config).to(device)
        dis = Discriminator().to(device)

        criterion_dis = nn.BCELoss().to(device)
        criterion_L2 = nn.MSELoss().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=model_config["lr"]
        )
        optimizer_GAN = torch.optim.Adam(dis.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=40, gamma=0.1
        )
        pck_50_overall_max = 0
        torch.cuda.empty_cache()
        for epoch in tqdm(range(model_config["epoch"])):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            model.train()
            total_loss = 0
            for idx, data in enumerate(data_loader["train"]):
                torch.cuda.empty_cache()
                optimizer_GAN.zero_grad()
                csi_data = data["input_wifi-csi"]
                csi_data = csi_data.clone().detach().to(device)
                bs = csi_data.size(0)

                keypoint = data["output"]

                xy_keypoint = keypoint[:, :, 0:2].to(device)
                confidence = keypoint[:, :, 2:3].to(device)

                (
                    correct_loss,
                    vq_loss,
                    z,
                    reconstructed_csi,
                    pred_xy_keypoint,
                ) = model(csi_data)
                # print(csi_data.size(), reconstructed_csi.size())
                ones_label = torch.ones(bs, 1).to(device)
                output_real = dis(csi_data)[0]
                errD_real = criterion_dis(output_real, ones_label)
                errD_real.backward(retain_graph=True)
                zeros_label = torch.zeros(bs, 1).to(device)
                output_fake = dis(reconstructed_csi)[0]
                errD_fake = criterion_dis(output_fake, zeros_label)
                errD_fake.backward(retain_graph=True)
                epsilon = 0.1
                num_iterations = 1
                alpha = epsilon / num_iterations
                csi_data.requires_grad = True
                for _ in range(num_iterations):
                    # Forward pass
                    outputs = model(csi_data)[3]
                    optimizer.zero_grad()

                    # Calculate the loss
                    # loss = criterion2(torch.mul(confidence, outputs), torch.mul(confidence, xy_keypoint)) / 32
                    loss = NMSELoss(csi_data, outputs)

                    # Zero all existing gradients

                    # Calculate gradients of model in backward pass
                    loss.backward(retain_graph=True)

                    # Collect datagrad
                    data_grad = csi_data.grad.data

                    # Collect the element-wise sign of the data gradient
                    sign_data_grad = data_grad.sign()

                    # Create the perturbed image by adjusting each pixel of the input image

                    perturbed_csi_data = csi_data + alpha * sign_data_grad

                    # Clip the perturbation to ensure it stays within the epsilon ball
                    # perturbed_csi_data = torch.clamp(perturbed_csi_data, 0, 1)

                    # Update the input for the next iteration
                    csi_data.data = (
                        perturbed_csi_data.detach()
                        .clone()
                        .requires_grad_(True)
                    )
                adv_csi = model(perturbed_csi_data)[3]
                zeros_label = torch.zeros(bs, 1).to(device)
                output_adv = dis(adv_csi)[0]
                errD_adv = criterion_dis(output_adv, zeros_label)
                errD_adv.backward(retain_graph=True)

                hpe_loss = (
                    criterion_L2(
                        torch.mul(confidence, pred_xy_keypoint),
                        torch.mul(confidence, xy_keypoint),
                    )
                    / 32
                )
                rec_loss = criterion_L2(reconstructed_csi, csi_data)

                loss = (
                    correct_loss
                    + vq_loss
                    + model_config["lambda"] * rec_loss
                    + hpe_loss
                )
                loss.backward(retain_graph=True)
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
                for idx, data in enumerate(data_loader["valid"]):
                    torch.cuda.empty_cache()
                    csi_data = data["input_wifi-csi"]
                    csi_data = csi_data.clone().detach().to(device)

                    keypoint = data["output"]
                    xy_keypoint = keypoint[:, :, 0:2].to(device)
                    confidence = keypoint[:, :, 2:3].to(device)

                    _, _, _, reconstructed_csi, pred_xy_keypoint = model(
                        csi_data
                    )
                    # print(csi_data.size(), reconstructed_csi.size())
                    # recontruction_loss = nmse(csi_data, reconstructed_csi)
                    # hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
                    # loss = model_config['lambda'] * recontruction_loss + hpe_loss

                    pred_xy_keypoint = pred_xy_keypoint.cpu()
                    xy_keypoint = xy_keypoint.cpu()
                    pred_xy_keypoint_pck = torch.transpose(
                        pred_xy_keypoint, 1, 2
                    )
                    xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)

                    metric.append(
                        calulate_error(pred_xy_keypoint, xy_keypoint)
                    )
                    pck_50_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.5
                        )
                    )
                    pck_20_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.2
                        )
                    )
                mean = np.mean(metric, 0) * 1000
                mpjpe_mean = mean[0]
                pa_mpjpe_mean = mean[1]
                pck_50 = np.mean(pck_50_iter, 0)
                pck_20 = np.mean(pck_20_iter, 0)
                pck_50_overall = pck_50[17]
                pck_20_overall = pck_20[17]
            if pck_50_overall > pck_50_overall_max:
                # print('saving the model at the end of epoch %d with pck_50: %.3f' % (epoch_index, pck_50_overall))
                torch.save(
                    model, os.path.join(checkpoint_folder, "best.pt")
                )
                pck_50_overall_max = pck_50_overall
                message = f"\nBest Epoch in epoch {epoch} with cookbook size is {model.vq.codebook_size} PCK50: {pck_50_overall_max}, PCK20: {pck_20_overall}"
            torch.save(model, os.path.join(checkpoint_folder, "last.pt"))
            scheduler.step()
        torch.cuda.empty_cache()
        print(message)
        for error_rate in np.arange(0.0, 1.0, 0.1):
            if error_rate != 0.0:
                continue
            print(
                f"with codebook size: {code_book_size} and {error_rate} in testing:"
            )
            model = torch.load(
                os.path.join(checkpoint_folder, "best.pt"),
                weights_only=False,
            )
            metric = []
            avg_nmse = []
            torch.cuda.empty_cache()
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for idx, data in enumerate(data_loader["test"]):
                    torch.cuda.empty_cache()
                    csi_data = data["input_wifi-csi"]
                    csi_data = csi_data.clone().detach().to(device)

                    keypoint = data["output"]
                    xy_keypoint = keypoint[:, :, 0:2].to(device)
                    confidence = keypoint[:, :, 2:3].to(device)

                    _, _, _, reconstructed_csi, pred_xy_keypoint = model(
                        csi_data, True, error_rate
                    )
                    # print(csi_data.size(), reconstructed_csi.size())
                    recontruction_loss = nmse(csi_data, reconstructed_csi)
                    # hpe_loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
                    # loss = _ + recontruction_loss + hpe_loss
                    avg_nmse.append(recontruction_loss.item())
                    pred_xy_keypoint = pred_xy_keypoint.cpu()
                    xy_keypoint = xy_keypoint.cpu()
                    pred_xy_keypoint_pck = torch.transpose(
                        pred_xy_keypoint, 1, 2
                    )
                    xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)

                    metric.append(
                        calulate_error(pred_xy_keypoint, xy_keypoint)
                    )
                    pck_50_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.5
                        )
                    )
                    pck_40_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.4
                        )
                    )
                    pck_30_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.3
                        )
                    )
                    pck_20_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.2
                        )
                    )
                    pck_10_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.1
                        )
                    )
                    pck_5_iter.append(
                        compute_pck_pckh(
                            pred_xy_keypoint_pck, xy_keypoint_pck, 0.05
                        )
                    )
            avg_nmse = np.mean(avg_nmse)
            mean = np.mean(metric, 0) * 1000
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
            print(
                "test result: nmse: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, pck_10: %.3f, pck_5: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f"
                % (
                    avg_nmse,
                    pck_50_overall,
                    pck_40_overall,
                    pck_30_overall,
                    pck_20_overall,
                    pck_10_overall,
                    pck_5_overall,
                    mpjpe_mean,
                    pa_mpjpe_mean,
                )
            )


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.manifold import TSNE

matplotlib.use("Agg")


def run_tsne(
    data_loader, model_config, model_checkpoint, device, image_folder_path
):
    # Load the model
    model = TSNECloudSense(model_config)
    model.load_state_dict(
        torch.load(model_checkpoint, weights_only=False).state_dict()
    )  # Load model weights
    model = model.to(device)
    model.eval()

    # Create a label map for the custom labels
    subject_map = {
        "S01": 0,
        "S02": 1,
        "S03": 2,
        "S04": 3,
        "S05": 4,
        "S06": 5,
        "S07": 6,
        "S08": 7,
        "S09": 8,
        "S10": 9,
        "S11": 10,
        "S12": 11,
        "S13": 12,
        "S14": 13,
        "S15": 14,
        "S16": 15,
        "S17": 16,
        "S18": 17,
        "S19": 18,
        "S20": 19,
        "S21": 20,
        "S22": 21,
        "S23": 22,
        "S24": 23,
        "S25": 24,
        "S26": 25,
        "S27": 26,
        "S28": 27,
        "S29": 28,
        "S30": 29,
        "S31": 30,
        "S32": 31,
        "S33": 32,
        "S34": 33,
        "S35": 34,
        "S36": 35,
        "S37": 36,
        "S38": 37,
        "S39": 38,
        "S40": 39,
    }

    action_map = {
        "A02": 0,
        "A03": 1,
        "A04": 2,
        "A05": 3,
        "A13": 4,
        "A14": 5,
        "A17": 6,
        "A18": 7,
        "A19": 8,
        "A20": 9,
        "A21": 10,
        "A22": 11,
        "A23": 12,
        "A27": 13,
    }

    action_targets = []  # Store action labels
    subject_targets = []  # Store subject labels
    test_embeddings = torch.zeros((0, 256 * 56))  # Store embeddings
    z_quantized_test_embeddings = torch.zeros((0, 256 * 56))  # Store embeddings

    # Extract embeddings and labels
    with torch.no_grad():
        for data in tqdm(data_loader["test"]):
            csi_data = data["input_wifi-csi"]
            csi_data = csi_data.clone().detach().to(device)

            # Forward pass through the model
            z, z_quantized = model(csi_data)

            # Collect action labels (e.g., "A02", "A03", ...)
            action = data["action"][0]
            action_targets.append(
                action_map[action]
            )  # Convert string label to numeric

            # Collect subject labels (if needed)
            subject_targets.append(subject_map[data["subject"][0]])

            # Flatten and append embeddings to the list
            test_embeddings = torch.cat(
                (
                    test_embeddings,
                    z.detach().cpu().flatten().unsqueeze(0),
                ),
                axis=0,
            )
            z_quantized_test_embeddings = torch.cat(
                (
                    z_quantized_test_embeddings,
                    z_quantized.detach().cpu().flatten().unsqueeze(0),
                ),
                axis=0,
            )

    # Convert action targets to numpy for plotting
    action_targets = np.array(action_targets)
    subject_targets = np.array(subject_targets)
    torch.save(test_embeddings, os.path.join(image_folder_path, "z.pt"))
    torch.save(z_quantized_test_embeddings, os.path.join(image_folder_path, "z_quantized.pt"))
    np.save(os.path.join(image_folder_path,"action.npy"), action_targets)
    np.save(os.path.join(image_folder_path, "subject.npy"), subject_targets)
    # Perform t-SNE on the embeddings
    tsne = TSNE(
        n_components=2,
        init="random",
        perplexity=26,
        metric="cosine",
        early_exaggeration=12,
        n_iter=1000,
    )
    tsne_proj = tsne.fit_transform(test_embeddings)

    # Plot t-SNE projection
    cmap = cm.get_cmap("tab20")  # Color map for discrete categories
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xticks([])
    ax.set_yticks([])
    # Scatter plot with color based on action targets
    scatter = ax.scatter(
        tsne_proj[:, 0],
        tsne_proj[:, 1],
        c=action_targets,
        cmap=cmap,
        alpha=0.75,
    )

    # Add color legend (showing label names)
    handles, labels = scatter.legend_elements()
    ax.legend(
        handles, list(action_map.keys()), title="Actions", fontsize="large"
    )

    # Optional: Add grid and title
    ax.grid(linestyle="--")

    # Show the plot
    plt.savefig(
        "/home/jackson-devworks/Desktop/CloudSense/tsne/z_action_tsne.jpg",
        dpi=500,
    )
