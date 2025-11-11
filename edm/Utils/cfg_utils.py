import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class multi_gaussian(nn.Module):
    def __init__(self, mean, covariance,low_rank):
        super(multi_gaussian, self).__init__()
        self.mean = mean.to(torch.float64)
        self.U, self.S, self.Vh = covariance
        self.U = self.U[:,:low_rank].clone().to(torch.float64)
        self.S = self.S[:low_rank].clone().to(torch.float64)
        self.Vh = self.Vh[:low_rank,:].clone().to(torch.float64)
    def forward(self, x, sigma):
        S = self.S/(self.S + sigma**2)
        I = torch.eye(self.U.shape[0], device=self.U.device, dtype=self.U.dtype)
        x = x.flatten(start_dim = 1)
        out = torch.mm(self.Vh, (x - self.mean).t())
        out = torch.mm(torch.diag(S),out)
        return (self.mean + torch.mm(self.U, torch.mm(torch.diag(S), torch.mm(self.Vh, (x - self.mean).t()))).t())

def generate_image(
    net, latents ,even_t=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, class_idx=None, use_second_order=False, device=torch.device('cuda')):
    sigma_list = []
    intermediates = []
    denoised_intermediates = []
    
    # Pick latents and labels.
    batch_size = latents.shape[0]
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    if class_idx is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if even_t == True:
        t_steps = torch.from_numpy(np.linspace(sigma_max, sigma_min, num_steps)).to(device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    #net.train()
    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        if use_second_order == True:
            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        intermediates.append(x_cur.detach().cpu())
        denoised_intermediates.append(denoised.detach().cpu())
        sigma_list.append(t_cur.item())
        
    # Save image grid.
    return sigma_list, intermediates, denoised_intermediates


def generate_image_Gaussian(
    net, Gaussian_net, latents ,even_t=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, device=torch.device('cuda')):
    sigma_list = []
    intermediates = []
    denoised_intermediates = []
    
    # Pick latents and labels.
    B, C, H, W = latents.shape
    batch_size = latents.shape[0]
    latents = latents.reshape(batch_size,-1)

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if even_t == True:
        t_steps = torch.from_numpy(np.linspace(sigma_max, sigma_min, num_steps)).to(device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = Gaussian_net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        '''
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        '''
        intermediates.append(x_cur.detach().cpu().reshape(B,C,H,W))
        denoised_intermediates.append(denoised.detach().cpu().reshape(B,C,H,W))
        sigma_list.append(t_cur.item())

    return sigma_list, intermediates, denoised_intermediates

def generate_image_Gaussian_cfg(
    net, Gaussian_net_cond, Gaussian_net_uncond, latents , guidance_strength=0, even_t=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, device=torch.device('cuda')):

    sigma_list = []
    intermediates = []
    denoised_intermediates_cond = []
    denoised_intermediates_uncond = []

    # Pick latents and labels.
    B, C, H, W = latents.shape
    batch_size = latents.shape[0]
    latents = latents.reshape(batch_size,-1)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if even_t == True:
        t_steps = torch.from_numpy(np.linspace(sigma_max, sigma_min, num_steps)).to(device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised_cond = Gaussian_net_cond(x_hat, t_hat).to(torch.float64)
        denoised_uncond = Gaussian_net_uncond(x_hat, t_hat).to(torch.float64)
        
        score_cond = (denoised_cond - x_hat) / t_hat
        score_uncond = (denoised_uncond - x_hat) / t_hat
        score_cfg = (1+guidance_strength)*score_cond - (guidance_strength)*score_uncond

        if t_cur <= 0.11:
            x_next = x_hat
        else:
            x_next = x_hat + (t_hat - t_next) * score_cfg # This implementation is correct

        '''
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        '''

        denoised_intermediates_cond.append(denoised_cond.detach().cpu().reshape(B,C,H,W))
        denoised_intermediates_uncond.append(denoised_uncond.detach().cpu().reshape(B,C,H,W))
        sigma_list.append(t_cur.item())

    # Save image grid.
    return sigma_list, intermediates, denoised_intermediates_cond, denoised_intermediates_uncond

'''
#################################################################################################################
# Applying CFG in EDM within a lmited interval
#################################################################################################################
'''
def generate_image_EDM_cfg_limited_interval(
    net_uncond, net_cond, latents , guidance_strength, sigma_high, sigma_low, main_model='cond', slack_scale=1, even_t=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, class_idx=None, device=torch.device('cuda')):
    
    sigma_list = []
    intermediates = []
    denoised_intermediates_cond = []
    denoised_intermediates_uncond = []

    # Pick latents and labels.
    batch_size = latents.shape[0]
    class_labels = None
    if net_cond.label_dim:
        class_labels = torch.eye(net_cond.label_dim, device=device)[torch.randint(net_cond.label_dim, size=[batch_size], device=device)]
    if class_idx is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1
        
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net_cond.sigma_min)
    sigma_max = min(sigma_max, net_cond.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if even_t == True:
        t_steps = torch.from_numpy(np.linspace(sigma_max, sigma_min, num_steps)).to(device)
    t_steps = torch.cat([net_cond.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net_cond.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised_cond = net_cond(x_hat, t_hat, class_labels).to(torch.float64)
        denoised_uncond = net_uncond(x_hat, t_hat, class_labels=None).to(torch.float64)
        score_cond = (denoised_cond - x_hat) / t_hat
        score_uncond = (denoised_uncond - x_hat) / t_hat
        guidance_direction = (denoised_cond - slack_scale*denoised_uncond)/t_hat

        score_cfg = score_cond + guidance_strength*guidance_direction

        if t_cur.item() <= sigma_high and t_cur.item() > sigma_low:
            x_next = x_hat + (t_hat - t_next) * score_cfg
        else:
            if main_model == 'cond':
                x_next = x_hat + (t_hat - t_next) * score_cond 
            else:
                x_next = x_hat + (t_hat - t_next) * score_uncond
        
        
        # Apply 2nd order correction.
        '''
        if i < num_steps - 1:
            denoised = net_cond(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            d_cur = - score_cond
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        '''
 
        intermediates.append(x_cur.detach().cpu())
        denoised_intermediates_cond.append(denoised_cond.detach().cpu())
        denoised_intermediates_uncond.append(denoised_uncond.detach().cpu())
        sigma_list.append(t_cur.item())
        
    # Save image grid.
    return sigma_list, intermediates, denoised_intermediates_cond, denoised_intermediates_uncond

'''
##############################################################################################################################################
modify a given reverse sampling trajectory at modified_timestep by adding modified_direction*modified_scale to the current intermediate sample
##############################################################################################################################################
'''
def modify_trajectory(net, num_steps, intermediates_list, modified_timestep, modified_scale, modified_direction, even_t=False, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, class_idx=None, device=torch.device('cuda')):

    sigma_list = []
    intermediates = []
    denoised_intermediates = []
    
    # Pick latents and labels.
    batch_size = len(intermediates_list[0])
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    if class_idx is not None:
        class_labels[:, :] = 0
        class_labels[:, class_idx] = 1

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, sigma_min)
    sigma_max = min(sigma_max, sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if even_t == True:
        t_steps = torch.from_numpy(np.linspace(sigma_max, sigma_min, num_steps)).to(device)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        if i < modified_timestep:
            continue
        elif i== modified_timestep:
            x_cur = intermediates_list[i].to(device) + modified_scale*modified_direction
        else: 
            x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        '''
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        '''
        intermediates.append(x_cur.detach().cpu())
        denoised_intermediates.append(denoised.detach().cpu())
        sigma_list.append(t_cur.item())
        
    # Save image grid.
    return sigma_list, intermediates, denoised_intermediates

'''
########################################################################################################################################
Obtain the corresponding unconditional denoised_image (denoised_uncond) from a given (conditional) noisy trajectory x_t, where t=T,...,1  
########################################################################################################################################
'''
def get_uncond_from_cond(net_uncond, intermediates_cond, sigma_list, num_steps=10, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, device=torch.device('cuda')):

    '''
    get unconditional score from a given conditional sampling trajectory 
    '''

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    num_steps = len(sigma_list)
    denoised_intermediates_uncond = []
    for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_t = intermediates_cond[i].to(device)

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_t + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_t)
        
        x_t_uncond = net_uncond(x_hat, torch.tensor(t_cur).to(device))

        denoised_intermediates_uncond.append(x_t_uncond.detach().cpu())
    return denoised_intermediates_uncond                   