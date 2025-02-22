import numpy as np
import torch
import os
import copy
import sys

from ann import ANN
from merton_main import DRM_Merton
#import int_ann # integral ANN
from iann import IDRM_ANN

INIT = False
dim = 5

def list2d_gen(diag, rest, dim):
    inner_list = [rest]*dim
    output = []
    for i in range(dim):
        tmp = list(inner_list)
        tmp[i] = diag
        output.append(tmp)
    return output

device = 'cuda' # cuda or cpu

# model paranmeters
xmax = 3.5
tmax = 1.0
r = 0.05
sigma = [0.5]*dim

rho = list2d_gen(1.0,0.,dim)
lam = 1.0
muj = [0.0]*dim

sigmaj = [0.5]*dim
rhoj = list2d_gen(1.0,0.,dim)

hg_levels = 0
scheme = 'bdf2'

merton = DRM_Merton(device, 0.02, tmax, xmax, r, sigma, rho, lam, muj, sigmaj, rhoj, hg_levels, scheme)
model = ANN(dim,1,64,device=device).to(device)
ann = IDRM_ANN(dim,1,64,device=device).to(device)
merton.iann = ann

def nlr(i):
    n_first = 2**11
    n_other = 2**11
    lr_first = 0.35e-3
    lr_other = 0.35e-3
    if i == 0:
        return n_first, lr_first
    else:
        return n_other, lr_other
    
if INIT:
    merton.init_fit(model,2**13, 2**16, lr=1e-3)
    exit(0)

# load the init model
RESUME_TIME = 0
RESUME_DIR = '15-20250124-2046'

# load the first two models obtained following the implicit-explicit euler scheme
pre_folder = './models/5-20250204-1342'
#pre_folder = './models/15-20250115-1824'
pre_nums = ['2','4']


if RESUME_TIME == 0:

    init_net_path = './init/dim_mod0' + str(dim)
    init_inet_path = './init/dim_iann0' + str(dim)
    
    init_net_path0 = pre_folder + '/model_' + pre_nums[0]
    init_net_path1 = pre_folder + '/model_' + pre_nums[1]
    
    init_inet_path = pre_folder  + '/iann_' + pre_nums[1]
    iann = torch.load(init_inet_path,map_location=torch.device(device))
    iann.device = device
    model0 = torch.load(init_net_path0,map_location=torch.device(device))
    model0.xmax = dim*xmax * torch.ones((1,1)).to(device)  
    Mfactor =  0.5*xmax*(1.+(3./dim)**0.5)
    model0.M = Mfactor*torch.ones((1,1)).to(device)
    model1 = torch.load(init_net_path1,map_location=torch.device(device))
    model1.xmax = dim*xmax* torch.ones((1,1)).to(device)
    model1.M = Mfactor*torch.ones((1,1)).to(device)
    model0.device = device
    model1.device = device
    merton.init_load_bdf2(model0, model1)
    merton.iann = iann
    del iann
    del model0
    del model1
    
else:
    init_net_path = './models/' + RESUME_DIR + '/model_' + str(RESUME_TIME)
    model = torch.load(init_net_path)
    merton.init_load_bdf2(model, RESUME_TIME)
    merton.date_str = RESUME_DIR
    del model

scriptname = os.path.basename(__file__)
os.system('mkdir -p ./models/' + merton.date_str)
os.system('cp ' + scriptname + ' ./models/' + merton.date_str) 


# training
i = RESUME_TIME
tau = merton.dt * (i+1)
while tau <= merton.tmax + 1.0*merton.dt:
    print('tau = ',tau)
    n, lr = nlr(i)
    merton.fit(2**12*dim, int(n), lr=lr, initial=False)
    tau += merton.dt
    i += 1



