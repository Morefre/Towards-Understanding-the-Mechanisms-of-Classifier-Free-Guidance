import torch
from torch.autograd.functional import jacobian



# Compute the Jacobian for each sample in the batch using a for loop
def compute_jacobian_batch(func, x_batch):
    B = len(x_batch)
    C, W, H = x_batch[0].shape

    # Create an empty list to store the Jacobians
    jacobians = []
    
    # Loop through each element in the batch and compute the Jacobian
    for i in range(x_batch.shape[0]):
        jac = jacobian(func, x_batch[i].unsqueeze(0))
        jacobians.append(jac.reshape(C*H*W, C*H*W))

    # Stack the list of Jacobians into a tensor
    return torch.stack(jacobians)