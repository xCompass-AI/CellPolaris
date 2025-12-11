import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)
    
class Multi_adv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(Multi_adv, self).__init__()
        self.domain_classifier = Discriminator(input_dim, hidden_dim, output_dim)
    
    def forward(self, inputs, labels):
        fea_lst = torch.zeros(1, inputs.size(1)).to(labels.device)
        idx_lst = torch.zeros(1).to(labels.device)
        for label in torch.unique(labels):
            idx = labels == label
            fea = inputs[idx, :]
            d_idx = torch.ones(len(fea)).to(labels.device) * label
            fea_lst = torch.cat((fea_lst, fea), dim=0)
            idx_lst = torch.cat((idx_lst, d_idx), dim=0)
        fea_lst = fea_lst[1:,:]
        idx_lst = idx_lst[1:].long().to(labels.device)

        fea = ReverseLayerF.apply(fea_lst, 1)
        domain_ids = idx_lst.view(-1)
        pred = self.domain_classifier(fea)
        loss = F.cross_entropy(pred, domain_ids)
        return loss


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=192, output_dim=1):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)