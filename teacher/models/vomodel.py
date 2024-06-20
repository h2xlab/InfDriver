from .flownet import FlowNetS
from .lstm import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fisher.fisher_utils import vmf_loss as fisher_NLL

class VOModel(nn.Module):
    def __init__(self, height, width):
        super(VOModel, self).__init__()
        '''
        Encoder
        '''
        self.encoder = FlowNetS()

        '''
        Feature shape for LSTM
        '''
        __tmp = torch.randn(1, 6, height, width)
        __tmp = self.encoder(__tmp)
   
        '''
        Decoder
        '''
        self.decoder = LSTM(int(np.prod(__tmp.size())))
    
    def forward(self, x):

        batch_size = x.size(0)
        x = self.encoder(x)

        return self.decoder(x.view(batch_size, -1))
    
    def get_loss(self, x, y):

        pose, A = self.forward(x)

        losses, pred_orth = fisher_NLL(A, y[:,6:], overreg=1.025)
        loss_A = losses.mean()

        '''
        Degree
        '''
        loss_p = torch.nn.functional.mse_loss(pose[:,:], y[:,:6])

        return loss_p, loss_A

    def step(self, x, y, optimizer):

        optimizer.zero_grad()
        loss_p, loss_A = self.get_loss(x, y)
        loss = loss_p + loss_A
        loss.backward()
        optimizer.step()

        return loss_p, loss_A