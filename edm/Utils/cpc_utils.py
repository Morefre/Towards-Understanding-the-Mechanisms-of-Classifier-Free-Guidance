import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import zipfile
import PIL.Image
import json
import dnnlib
import training
import pickle
from tqdm import tqdm
# from scipy.optimize import linear_sum_assignment
import training.loss
try:
    import pyspng
except ImportError:
    pyspng = None
import matplotlib.pyplot as plt
import argparse
import h5py
#from U_Net_models.skip import skip
import copy
import argparse
#from train_InDI import *

#######################################################
# Apply linear CFG to a EDM model
#######################################################
def generate_image_Gaussian_cpc_limited_interval(
    net_EDM, Gaussian_net_cond, Gaussian_net_uncond, latents , sigma_high, sigma_low, guide_type = 'pos', guidance_strength=0, even_t=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0, class_idx=None, device=torch.device('cuda')):
    
    '''
    guide_type: "pos", "neg", "mean_shift, normal"
    '''

    mean_cond = Gaussian_net_cond.mean
    mean_cond = mean_cond.to(device)
    mean_uncond = Gaussian_net_uncond.mean
    mean_uncond = mean_uncond.to(device)
    mean_shift = mean_cond - mean_uncond
    mean_shift = mean_shift.to(device)

    sigma_list = []
    intermediates = []
    denoised_intermediates_cond = []
    
    # Pick latents and labels.
    B, C, H, W = latents.shape
    batch_size = latents.shape[0]
    latents = latents.reshape(batch_size,-1)
    class_labels = None
    if net_EDM.label_dim:
        class_labels = torch.eye(net_EDM.label_dim, device=device)[torch.randint(net_EDM.label_dim, size=[batch_size], device=device)]
    
    if class_idx is not None and class_labels is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net_EDM.sigma_min)
    sigma_max = min(sigma_max, net_EDM.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if even_t == True:
        t_steps = torch.from_numpy(np.linspace(sigma_max, sigma_min, num_steps)).to(device)
    t_steps = torch.cat([net_EDM.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net_EDM.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        U_c = Gaussian_net_cond.U
        S_c = Gaussian_net_cond.S
        Vh_c = Gaussian_net_cond.Vh
        cov_cond = U_c@torch.diag(S_c/(S_c+t_cur**2))@Vh_c


        U_uc = Gaussian_net_uncond.U
        S_uc = Gaussian_net_uncond.S
        Vh_uc = Gaussian_net_uncond.Vh
        cov_uncond = U_uc@torch.diag(S_uc/(S_uc+t_cur**2))@Vh_uc

        if guide_type == 'mean_shift':
            guidance_direction = (mean_shift - cov_uncond@mean_shift) / t_hat
            guidance_direction = guidance_direction.reshape(1,C,H,W)

        if guide_type == 'normal':
            denoised_cond = Gaussian_net_cond(x_hat, t_hat).to(torch.float64)
            denoised_uncond = Gaussian_net_uncond(x_hat, t_hat).to(torch.float64)
            guidance_direction = (denoised_cond - denoised_uncond) / t_hat
            guidance_direction = guidance_direction.reshape(B,C,H,W)

        if guide_type == 'pos':
            S_cpc, U_cpc = torch.linalg.eigh(cov_cond - cov_uncond)
            S_cpc_pos = torch.tensor(S_cpc).to(device)
            S_cpc_pos[S_cpc_pos<0] = 0
            jacobian_cpc_pos = U_cpc@torch.diag(S_cpc_pos)@U_cpc.T
            jacobian_cpc_pos_batched = jacobian_cpc_pos.unsqueeze(0).expand(B,C*H*W,C*H*W)
            guidance_direction = torch.bmm(jacobian_cpc_pos_batched,x_cur.reshape(B,C*H*W,1)-mean_cond.reshape(C*H*W,1)) / t_hat
            guidance_direction = guidance_direction.reshape(B,C,H,W)

        if guide_type == 'neg':
            S_cpc, U_cpc = torch.linalg.eigh(cov_cond - cov_uncond)
            S_cpc_neg = torch.tensor(S_cpc).to(device)
            S_cpc_neg[S_cpc_neg>0] = 0
            jacobian_cpc_neg = U_cpc@torch.diag(S_cpc_neg)@U_cpc.T
            jacobian_cpc_neg_batched = jacobian_cpc_neg.unsqueeze(0).expand(B,C*H*W,C*H*W)
            guidance_direction = torch.bmm(jacobian_cpc_neg_batched,x_cur.reshape(B,C*H*W,1)-mean_cond.reshape(C*H*W,1)) / t_hat
            guidance_direction = guidance_direction.reshape(B,C,H,W)

        if guide_type == 'cpc':
            jacobian_cpc = (cov_cond - cov_uncond).to(torch.float64)
            jacobian_cpc_batched = jacobian_cpc.unsqueeze(0).expand(B,C*H*W,C*H*W)
            guidance_direction = torch.bmm(jacobian_cpc_batched,x_cur.reshape(B,C*H*W,1)-mean_cond.reshape(C*H*W,1)) / t_hat
            guidance_direction = guidance_direction.reshape(B,C,H,W)

            denoised_cond = Gaussian_net_cond(x_hat, t_hat).to(torch.float64)

        guidance_direction = guidance_direction.to(device)
        
        x_hat = x_hat.reshape(B,C,H,W)
        denoised_EDM = net_EDM(x_hat, t_hat, class_labels).to(torch.float64)
        score_net_EDM = (denoised_EDM - x_hat) / t_hat

        score_cfg = score_net_EDM + guidance_strength*guidance_direction
       
        # Euler step.
        if t_cur.item() <= sigma_high and t_cur.item() > sigma_low:
            x_next = x_hat + (t_hat - t_next)*score_cfg
        else:
            x_next = x_hat + (t_hat - t_next) * score_net_EDM # This implementation is correct

        '''
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        '''
       
        intermediates.append(x_cur.detach().cpu().reshape(B,C,H,W))
        denoised_intermediates_cond.append(denoised_EDM.detach().cpu().reshape(B,C,H,W))
        sigma_list.append(t_cur.item())

    return sigma_list, intermediates, denoised_intermediates_cond


#######################################################################################################
# Apply linear CFG to a linear model. The EDM model is used only for constructing the sampling schedule
#######################################################################################################
def generate_image_pure_Gaussian_cpc_limited_interval(
    net_EDM, Gaussian_net_cond, Gaussian_net_uncond, latents , sigma_high, sigma_low, 
    guide_type = 'pos', guidance_strength=0, even_t=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0, class_idx=None, device=torch.device('cuda')):

    #guide_type: "pos", "neg", "mean_shift, normal"
    # pos: guiding with positive CPCs
    # neg: guiding with negative CPCs
    # mean_shift: guiding with mean shift guidance
    # cpc: guiding with both positive and negative CPCs
    # normal: the standard CFG guidance
    mean_cond = Gaussian_net_cond.mean
    mean_cond = mean_cond.to(device)
    mean_uncond = Gaussian_net_uncond.mean
    mean_uncond = mean_uncond.to(device)
    mean_shift = mean_cond - mean_uncond
    mean_shift = mean_shift.to(device)

    sigma_list = []
    intermediates = []
    denoised_intermediates_cond = []
    # denoised_intermediates_uncond = []

    # Pick latents and labels.
    B, C, H, W = latents.shape
    batch_size = latents.shape[0]
    latents = latents.reshape(batch_size,-1)
    class_labels = None
    if net_EDM.label_dim:
        class_labels = torch.eye(net_EDM.label_dim, device=device)[torch.randint(net_EDM.label_dim, size=[batch_size], device=device)]
    
    if class_idx is not None and class_labels is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net_EDM.sigma_min)
    sigma_max = min(sigma_max, net_EDM.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if even_t == True:
        t_steps = torch.from_numpy(np.linspace(sigma_max, sigma_min, num_steps)).to(device)
    t_steps = torch.cat([net_EDM.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net_EDM.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        U_c = Gaussian_net_cond.U
        S_c = Gaussian_net_cond.S
        Vh_c = Gaussian_net_cond.Vh
        cov_cond = U_c@torch.diag(S_c/(S_c+t_cur**2))@Vh_c


        U_uc = Gaussian_net_uncond.U
        S_uc = Gaussian_net_uncond.S
        Vh_uc = Gaussian_net_uncond.Vh
        cov_uncond = U_uc@torch.diag(S_uc/(S_uc+t_cur**2))@Vh_uc

        if guide_type == 'mean_shift':
            U_uc = Gaussian_net_uncond.U
            S_uc = Gaussian_net_uncond.S
            Vh_uc = Gaussian_net_uncond.Vh
            cov_uncond = U_uc@torch.diag(S_uc/(S_uc+t_cur**2))@Vh_uc
            cov_uncond = cov_uncond.to(latents.device)
            guidance_direction = (mean_shift - cov_uncond@mean_shift) / t_hat
            guidance_direction = guidance_direction.reshape(1,C,H,W)

        if guide_type == 'normal':
            denoised_cond = Gaussian_net_cond(x_hat, t_hat).to(torch.float64)
            denoised_uncond = Gaussian_net_uncond(x_hat, t_hat).to(torch.float64)
            guidance_direction = (denoised_cond - denoised_uncond) / t_hat

        if guide_type == 'pos':
            S_cpc, U_cpc = torch.linalg.eigh(cov_cond - cov_uncond)
            S_cpc_pos = torch.tensor(S_cpc).to(device)
            S_cpc_pos[S_cpc_pos<0] = 0
            jacobian_cpc_pos = U_cpc@torch.diag(S_cpc_pos)@U_cpc.T
            jacobian_cpc_pos_batched = jacobian_cpc_pos.unsqueeze(0).expand(B,C*H*W,C*H*W)
            guidance_direction = torch.bmm(jacobian_cpc_pos_batched,x_cur.reshape(B,C*H*W,1)-mean_cond.reshape(C*H*W,1)) / t_hat
            guidance_direction = guidance_direction.reshape(B,C,H,W)

        if guide_type == 'neg':
            S_cpc, U_cpc = torch.linalg.eigh(cov_cond - cov_uncond)
            S_cpc_neg = torch.tensor(S_cpc).to(device)
            S_cpc_neg[S_cpc_neg>0] = 0
            jacobian_cpc_neg = U_cpc@torch.diag(S_cpc_neg)@U_cpc.T
            jacobian_cpc_neg_batched = jacobian_cpc_neg.unsqueeze(0).expand(B,C*H*W,C*H*W)
            guidance_direction = torch.bmm(jacobian_cpc_neg_batched,x_cur.reshape(B,C*H*W,1)-mean_cond.reshape(C*H*W,1)) / t_hat
            guidance_direction = guidance_direction.reshape(B,C,H,W)


        if guide_type == 'cpc':
            jacobian_cpc = (cov_cond - cov_uncond).to(torch.float64)
            jacobian_cpc_batched = jacobian_cpc.unsqueeze(0).expand(B,C*H*W,C*H*W)
            guidance_direction = torch.bmm(jacobian_cpc_batched,x_cur.reshape(B,C*H*W,1)-mean_cond.reshape(C*H*W,1)) / t_hat
            guidance_direction = guidance_direction.reshape(B,C,H,W)

            denoised_cond = Gaussian_net_cond(x_hat, t_hat).to(torch.float64)

        guidance_direction = guidance_direction.to(device)
        
        x_hat = x_hat.reshape(B,C,H,W)
        denoised_cond = Gaussian_net_cond(x_hat, t_hat).to(torch.float64).reshape(B,C,H,W)
        score_net_Gaussian = (denoised_cond - x_hat) / t_hat

        score_cfg = score_net_Gaussian + guidance_strength*guidance_direction
       
        # Euler step.
        if t_cur.item() <= sigma_high and t_cur.item() > sigma_low:
            x_next = x_hat + (t_hat - t_next)*score_cfg
        else:
            x_next = x_hat + (t_hat - t_next) * score_net_Gaussian # This implementation is correct
        '''
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        '''
       
        intermediates.append(x_cur.detach().cpu().reshape(B,C,H,W))
        denoised_intermediates_cond.append(denoised_cond.detach().cpu().reshape(B,C,H,W))
        sigma_list.append(t_cur.item())

    return sigma_list, intermediates, denoised_intermediates_cond