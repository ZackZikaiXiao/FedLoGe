import torch

import os
import copy
import numpy as np
import random
import torch
import copy
import pdb
import torch.nn as nn
from tqdm import tqdm
from options import args_parser, args_parser_cifar10
from util.update_baseline import LocalUpdate, globaltest, localtest, globaltest_classmean, globaltest_calibra, globaltest_feat_collapse
from util.fedavg import *
import torch.optim as optim
# from util.util import add_noise
from util.dataset import *
from model.build_model import build_model
from util.dispatch import *
from util.losses import *
from util.etf_methods import ETF_Classifier
import matplotlib.pyplot as plt
import seaborn as sns

def angle_between(v1, v2):
    cos_angle = (v1.dot(v2)) / (torch.norm(v1) * torch.norm(v2))
    angle = torch.acos(cos_angle)  # in radians
    degrees = angle * 180 / np.pi
    return degrees  # if you want degrees you can do: angle * 180 / np.pi

class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

        self.LWS = LWS
        self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x

    def gen_sparse_ETF(self, feat_in=512, num_classes=100, beta=0.6):
        # Initialize ETF
        etf = copy.deepcopy(self.ori_M)
        # Sparsify ETF
        num_zero_elements = int(beta * feat_in * num_classes)
        zero_indices = np.random.choice(feat_in * num_classes, num_zero_elements, replace=False)
        etf_flatten = etf.flatten()
        etf_flatten[zero_indices] = 0
        sparse_etf = etf_flatten.reshape(feat_in, num_classes)
        
        # Adjust non-zero elements
        sparse_etf = torch.tensor(sparse_etf, requires_grad=True)
        
        
        # Create a mask where the initial tensor is non-zero
        mask = (sparse_etf != 0).float()

        # Optimizer
        optimizer = optim.Adam([sparse_etf], lr=0.01)

        # Number of optimization steps
        n_steps = 10000

        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Constraint 1: L2 norm of each row should be 1
            row_norms = torch.norm(sparse_etf, p=2, dim=0)
            norm_loss = torch.sum((row_norms - 1)**2)
            
            # Constraint 2: Maximize the angle between vectors (minimize cosine similarity)
            normalized_etf = sparse_etf / row_norms
            cos_sim = torch.mm(normalized_etf.t(), normalized_etf)
            torch.diagonal(cos_sim).fill_(-1)
            angle_loss = -torch.acos(cos_sim.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
            # angle_loss = -torch.sum(torch.acos(torch.clamp(cos_sim, -0.9999999, 0.9999999)))
        
            
            # Total loss
            loss = norm_loss + angle_loss
            
            # Backpropagation
            loss.backward()
            
            
            # Apply the mask to the gradients
            if sparse_etf.grad is not None:
                sparse_etf.grad *= mask
                
                
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step}, Loss {loss.item()}")
        return sparse_etf



etf = ETF_Classifier(512, 100)
sparse_etf = etf.gen_sparse_ETF()


import numpy as np
import torch

# Normalize each column to have L2 norm = 1
col_norms = torch.norm(sparse_etf, p=2, dim=0, keepdim=True)
normalized_etf = sparse_etf / col_norms

# Compute cosine similarities
cosine_similarities = torch.mm(normalized_etf.t(), normalized_etf)

# Zero out the diagonal (we don't want to compare vectors with themselves)
torch.diagonal(cosine_similarities).fill_(float('nan'))

# Compute angles in radians
angles_radians = torch.acos(torch.clamp(cosine_similarities, -1, 1))

# Convert angles from radians to degrees
angles_degrees = angles_radians * (180 / np.pi)

# Convert to numpy array
angles_degrees_numpy = angles_degrees.cpu().detach().numpy()

# Calculate mean and variance of angles, ignoring NaNs
angle_mean = np.nanmean(angles_degrees_numpy)
angle_variance = np.nanvar(angles_degrees_numpy)

# Calculate mean and variance of norms
col_norms_numpy = col_norms.cpu().detach().numpy()
norm_mean = np.mean(col_norms_numpy)
norm_variance = np.var(col_norms_numpy)

print(f"Angle Mean: {angle_mean}, Angle Variance: {angle_variance}")
print(f"Norm Mean: {norm_mean}, Norm Variance: {norm_variance}")






# angle_between(sparse_etf[:,0], sparse_etf[:,90])

# 可视化
angles_degrees_numpy = torch.acos(torch.clamp(cos_sim, -0.9999999, 0.9999999)) * (180 / np.pi)
sns.heatmap(angles_degrees_numpy.cpu().detach().numpy(), cmap='coolwarm')
plt.savefig('angle_matrix_heatmap.png')