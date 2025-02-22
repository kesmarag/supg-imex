import torch
import torch.nn as nn
import numpy as np

class IDRM_ANN(nn.Module):    
    def __init__(self, n_input, n_output, n_hidden, device='cpu'):
        super(IDRM_ANN, self).__init__()
        self.device = device
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.w0 = nn.Linear(n_input,n_hidden,bias=True)
        self.w0.weight = nn.Parameter(torch.ones((n_hidden,n_input))) # weight
        # first DGM layer tensors
        self.ug1 = nn.Linear(n_input,n_hidden,bias=False)
        self.wg1 = nn.Linear(n_hidden,n_hidden)
        self.uz1 = nn.Linear(n_input,n_hidden,bias=False)
        self.wz1 = nn.Linear(n_hidden,n_hidden)
        self.ur1 = nn.Linear(n_input,n_hidden,bias=False)
        self.wr1 = nn.Linear(n_hidden,n_hidden)
        self.uh1 = nn.Linear(n_input,n_hidden,bias=False)
        self.wh1 = nn.Linear(n_hidden,n_hidden)

        # first DMG layer - initialization
        nn.init.xavier_uniform_(self.ug1.weight)
        nn.init.xavier_uniform_(self.wg1.weight)
        nn.init.xavier_uniform_(self.uz1.weight)
        nn.init.xavier_uniform_(self.wz1.weight)
        nn.init.xavier_uniform_(self.ur1.weight)
        nn.init.xavier_uniform_(self.wr1.weight)
        nn.init.xavier_uniform_(self.uh1.weight)
        nn.init.xavier_uniform_(self.wh1.weight)

        # second DGM layer tensors
        self.ug2 = nn.Linear(n_input,n_hidden,bias=False)
        self.wg2 = nn.Linear(n_hidden,n_hidden)
        self.uz2 = nn.Linear(n_input,n_hidden,bias=False)
        self.wz2 = nn.Linear(n_hidden,n_hidden)
        self.ur2 = nn.Linear(n_input,n_hidden,bias=False)
        self.wr2 = nn.Linear(n_hidden,n_hidden)
        self.uh2 = nn.Linear(n_input,n_hidden,bias=False)
        self.wh2 = nn.Linear(n_hidden,n_hidden)

        # second DMG layer - initialization
        nn.init.xavier_uniform_(self.ug2.weight)
        nn.init.xavier_uniform_(self.wg2.weight)
        nn.init.xavier_uniform_(self.uz2.weight)
        nn.init.xavier_uniform_(self.wz2.weight)
        nn.init.xavier_uniform_(self.ur2.weight)
        nn.init.xavier_uniform_(self.wr2.weight)
        nn.init.xavier_uniform_(self.uh2.weight)
        nn.init.xavier_uniform_(self.wh2.weight)
        
        self.w4 = nn.Linear(n_hidden,n_output,bias=False)
        torch.nn.init.xavier_uniform_(self.w4.weight)

        
    def _forward_main(self, x):
        x = torch.abs(x) + 0.1
        #x_m_1 = x + 1.0 # the maximum is at (1.0,...,1.0)^T
        afunc = torch.tanh
        # input layer
        s0 = afunc(self.w0(x))
        # first DGL
        g1 = afunc(self.ug1(x) + self.wg1(s0))
        z1 = afunc(self.uz1(x) + self.wz1(s0))
        r1 = afunc(self.ur1(x) + self.wr1(s0))
        h1 = afunc(self.uh1(x) + self.wh1(torch.mul(s0, r1)))
        s1 = torch.mul((1.0 - g1), h1) + torch.mul(z1, s0)
        # second DGL
        g2 = afunc(self.ug2(x) + self.wg2(s1))
        z2 = afunc(self.uz2(x) + self.wz2(s1))
        r2 = afunc(self.ur2(x) + self.wr2(s1))
        h2 = afunc(self.uh2(x) + self.wh2(torch.mul(s1, r2)))
        s2 = torch.mul((1.0 - g2), h2) + torch.mul(z2, s1)
        # print(x.mean(axis=1))
        y = self.w4(s1)
        return y
            

    def forward(self, x):
        output = self._forward_main(x)
        return output
