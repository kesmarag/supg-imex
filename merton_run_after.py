import numpy as np
import torch

import os
import copy
import sys
# sys.path.insert(1, '/home/kesmarag/Git/dgm')

from ann import ANN
from merton_main import DRM_Merton
#import int_ann # integral ANN
from iann import IDRM_ANN
# init


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

device = 'cpu' # cuda or cpu

# model paranmeters
xmax = 3.5
tmax = 1.0
r = 0.05
sigma = [0.5]*dim

rho = list2d_gen(1.0,0.5,dim)
lam = 1.0
muj = [-0.**3]*dim

sigmaj = [0.5]*dim
rhoj = list2d_gen(1.0,0.2,dim)

hg_levels = 0
scheme = 'euler'

merton = DRM_Merton(device, 0.01, tmax, xmax, r, sigma, rho, lam, muj, sigmaj, rhoj, hg_levels, scheme)
model = ANN(dim,1,64,device=device).to(device)
#ann = int_ann.INT_ANN(dim,1,64).to(device)
ann = IDRM_ANN(dim,1,64,device=device).to(device)
merton.iann = ann


#merton.init_iann_fit(ann,2**13, 2**15, lr=1e-3)
#exit(0)
def nlr(i):
    n_first = 2**11
    n_other = 2**11
    lr_first = 3e-4
    lr_other = 3e-4
    if i == 0:
        return n_first, lr_first
    else:
        return n_other, lr_other
    
if INIT:
    merton.init_fit(model,2**13, 2**15, lr=1e-3)
    exit(0)

# load the init model
RESUME_TIME = 0
RESUME_DIR = '15-20250127-0708'
if RESUME_TIME == 0:
    init_net_path = './init/dim_mod0' + str(dim)
    init_inet_path = './init/dim_iann0' + str(dim)
    model = torch.load(init_net_path,map_location=torch.device(device))
    model.xmax = xmax* dim * torch.ones((1,1)).to(device)
    Mfactor =  0.5*xmax*(1.+(3./dim)**0.5)
    model.M = Mfactor*torch.ones((1,1)).to(device)
    iann = torch.load(init_inet_path,map_location=torch.device(device))
    iann.device=device
    model.device=device
    merton.init_load_euler(model)
    merton.iann = iann
    del iann
    del model
else:
    init_net_path = './models/' + RESUME_DIR + '/model_' + str(RESUME_TIME)
    init_inet_path = './models/' + RESUME_DIR + '/iann_' + str(RESUME_TIME)
    model = torch.load(init_net_path,map_location=torch.device(device))
    model.xmax = xmax* dim * torch.ones((1,1)).to(device)
    Mfactor = 0.5*xmax*(1.+(3./dim)**0.5)
    model.M = Mfactor*torch.ones((1,1)).to(device)
    model.device=device
    merton.init_load_euler(model, RESUME_TIME)
    iann = torch.load(init_inet_path,map_location=torch.device(device))
    iann.device=device    
    merton.iann = iann
    merton.date_str = RESUME_DIR
    del model
    del iann

# save the string filename to the result folder 
scriptname = os.path.basename(__file__)
os.system('mkdir -p ./models/' + merton.date_str)
os.system('cp ' + scriptname + ' ./models/' + merton.date_str) 
os.system('cp ann.py' +  ' ./models/' + merton.date_str)
os.system('cp iann.py' + ' ./models/' + merton.date_str) 
os.system('cp merton_main.py' + ' ./models/' + merton.date_str) 
os.system('cp ./init/dim_mod0' + str(dim) + ' ./models/' + merton.date_str) 
os.system('cp ./init/dim_iann0' + str(dim) + ' ./models/' + merton.date_str) 
# training
i = RESUME_TIME
tau = merton.dt * (i+1)
while tau <= merton.tmax + 6.0*merton.dt:
    print('tau = ',tau)
    n, lr = nlr(i)
    merton.fit(2**12*dim, int(n), lr=lr, initial=False)
    tau += merton.dt
    i += 1


    

