import torch
import torch.nn as nn
import numpy as np

class ANN(nn.Module):    
    def __init__(self, n_input, n_output, n_hidden, xmax=3*3.5, xp=3.0, r=0.05, t=0.0, device='cpu'):
        super(ANN, self).__init__()
        self.device = device
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.r = r
        self.t = t
        self.xmax = xmax * torch.ones((1,self.n_output)).to(device)
        self.M = xp * torch.ones((1,self.n_output))
        self.M = self.M.to(self.device)
        self.xp =  xp * torch.ones((1,self.n_output))
        self.xp = self.xp.to(self.device)
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
        xx = torch.abs(x)-1. 
        #x_m_1 = x - 1.0 # the maximum is at (1.0,...,1.0)^T
        afunc = torch.tanh
        # input layer
        s0 = afunc(self.w0(xx))
        # first DGL
        g1 = afunc(self.ug1(xx) + self.wg1(s0))
        z1 = afunc(self.uz1(xx) + self.wz1(s0))
        r1 = afunc(self.ur1(xx) + self.wr1(s0))
        h1 = afunc(self.uh1(xx) + self.wh1(torch.mul(s0, r1)))
        s1 = torch.mul((1.0 - g1), h1) + torch.mul(z1, s0)
        # second DGL
        g2 = afunc(self.ug2(xx) + self.wg2(s1))
        z2 = afunc(self.uz2(xx) + self.wz2(s1))
        r2 = afunc(self.ur2(xx) + self.wr2(s1))
        h2 = afunc(self.uh2(xx) + self.wh2(torch.mul(s1, r2)))
        s2 = torch.mul((1.0 - g2), h2) + torch.mul(z2, s1)

        m = nn.Softplus(beta=0.5)

        y = 0.05*m(self.w4(s2))
        
        
        if self.t==0.0:
            return self._payoff_and_sigmoid(x)
        elif self.t==-1.0: # just the stochastic part
            return y
        else:
            return y + self._payoff_and_sigmoid(x)
            
    def _payoff_and_sigmoid(self, x):
        #w = 1.0/self.n_input
        #sval = 0.0
        #for i in range(self.n_input):
        #    sval += w * torch.abs(x[:,i].reshape(-1,1))
        sval = (torch.abs(x)).mean(-1).reshape(-1,1) 
        splus = torch.nn.functional.relu(sval-1.).to(self.device)
        tb = (1-np.exp(-self.r*self.t))*torch.sigmoid(5.0*sval-5.0*np.exp(-self.r*self.t)).to(self.device)
        lb = splus + tb
        return lb
    

    def forward(self,x):
        if self.t == -1.0:
            return self._forward_main(x)
        xmax = torch.abs(x).max(axis=-1)[0]
        xmu = torch.abs(x).mean(axis=-1)
        criterion = torch.abs(x).max(axis=-1)[0] > torch.max(xmu, self.M) * self.xmax/self.M
        q = torch.where(criterion, self.xmax/xmax, self.M/torch.max(xmu,self.M)).T
        f = self._forward_main(q*x)
        x_diff = (torch.abs(x) - q*torch.abs(x)).mean(axis=-1).reshape(-1,1)
        return f + x_diff
        
