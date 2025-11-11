import torch
from scipy.optimize import linear_sum_assignment
import numpy as np


def normalize_image(img, clip=False):
    if clip == True:
        img = (img * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()
    else:
        img = img-torch.min(img)
        img = img/torch.max(img)
    return img

def normalize_np(a):
    a = a-np.min(a)
    a = a/np.max(a)
    return a

def get_image_grid(denoised_intermediates, num_row, num_col, resolution, clip=False):
    image_grid = denoised_intermediates
    image_grid = normalize_image(image_grid, clip=clip)
    image_grid = image_grid.reshape(num_row, num_col, *image_grid.shape[1:]).permute(0, 3, 1, 4, 2)
    image_grid = image_grid.reshape(num_row*resolution, num_col*resolution, 3)
    return image_grid


def find_optimal_with_sign(A,B):
    # This function is used to pair singular vectors of A and B. 
    # A,B both have size mxn, where m is the dimension of a feature and n is the total number of the features
    # A,B are torch tensors
    similarity_pos = A.T@B
    similarity_neg = -A.T@B
    similarity_pos[similarity_pos<similarity_neg] = similarity_neg[similarity_pos<similarity_neg]
    similarity_matrix = similarity_pos.detach().cpu().numpy()
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    return row_indices, col_indices, similarity_matrix