import torch
import torch.nn as nn
import numpy as np
import math
import time
import copy
import os
import iann
from datetime import datetime
np.random.seed(10)

# 2 optimization schemes - Euler and BDF-2
# 2 integration modes - GH (gh_levels>0) and using ANN (gh_levels=0)


class DRM_Merton():
    def __init__(self, device, dt, tmax, xmax, r, sigma, rho, lam, muj, sigmaj, rhoj, gh_levels, scheme='Euler'):
        self.dim = len(sigma)
        self.dt = dt
        self.tmax = tmax
        self.xmax = xmax
        self.r = r
        self.sigma = sigma
        self.rho = rho
        self.lam = lam
        self.iann = None
        self.muj = muj
        self._sigmaj = sigmaj
        self._rhoj = rhoj
        self.sigmaj = self._cov_gen(sigmaj,rhoj,self.dim)
        self.x = None
        self.tx = None
        self.device = device
        self.time_step = 0
        if scheme == 'bdf2':
            self.time_step = 2
        self.date_dir = None
        self.gh_levels = gh_levels
        self.pdf = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.tensor(np.array(muj)), covariance_matrix=torch.tensor(np.array(self.sigmaj)))
        self.date_str = self._folder_name()
        self.scheme = scheme
        print(self.date_str)

    def _folder_name(self):
        dt = datetime.now().strftime('%Y%m%d-%H%M')
        return str(self.dim)+ '-'  + dt
    
    def _cov_gen(self, sigma, rho, dim):
        row_lst = []
        for i in range(dim):
           col_lst = []
           for j in range(dim):
               col_lst.append(sigma[i]*sigma[j]*rho[i][j])
           row_lst.append(col_lst)
        return row_lst
        
    def _sampling_init(self, n, alpha = 1., xmin=0., xmax=3.): # using Dirichlet samplin
        dirichlet_dist = torch.distributions.Dirichlet(alpha*torch.ones(self.dim))
        samples = dirichlet_dist.sample((n,))
        # a = 0.5*torch.ones_like(samples)
        # c = 2.*torch.bernoulli(a)-1.
        # d = torch.rand(c.shape)
        # x = samples[:,:-1]
        r = 3.0
        return (r*self.dim*samples).requires_grad_().to(self.device) 

        
    def _sampling(self,n):
        soboleng_yy = torch.quasirandom.SobolEngine(dimension=self.dim,seed=np.random.randint(1000),scramble=True)
           
        y = self.xmax*soboleng_yy.draw(n)
        
        #r = 0.8 + 0.4*torch.rand(y.shape)
        
        #x = r*y
        
        # a = 0.5*torch.ones_like(y)
        # c = 2.*torch.bernoulli(a)-1.
        # z = y

        return y.requires_grad_().to(self.device) # 
    
    def _samplingS(self,n):
        simplex = self._sampling_simplex(n//2)
        rectangle = self._sampling_rectangle(n//2)
        x = torch.cat((simplex,rectangle),0)
        return x
        
    
    def _thsampling(self,n, alpha=2.):
        dirichlet_dist = torch.distributions.Dirichlet(alpha*torch.ones(self.dim))
        samples = dirichlet_dist.sample((n,))
        a = 0.5*torch.ones_like(samples)
        c = 2.*torch.bernoulli(a)-1.
        d = torch.rand(c.shape)
        # x = samples[:,:-1]
        r = 3.*d
        return (r*self.dim*samples).requires_grad_().to(self.device) 


    def _icost_function_q_euler(self,n,q=1): # loss function for the automatic integration
        self.x = self._sampling(n)
        N = self.x.shape[0]
        df = torch.zeros((N,1)).to(self.device)
        # rn = 0.8 + 0.4*torch.rand(N,1).to(self.device)
        for j in range(q):
            z = self.pdf.sample((N,)).float().to(self.device) # we have set σ = 0.5 ###################################
            xez = self.x*torch.exp(z)
            fxez = self.model_prev(xez)
            fx = self.model_prev(self.x.reshape(-1,self.dim))
            df += (fxez - fx).reshape(-1,1)
        return (self.iann(self.x)-1./q*df.reshape(-1,1))**2 #+ self.iann(0.01*self.x)**2


    def _icost_function_q_init(self,x,q=1): # loss function for the automatic integration
        N = x.shape[0]
        df = torch.zeros((N,1)).to(self.device)
        def payoff(x):
            #print(x.mean(axis=1).shape)
            return torch.maximum(torch.abs(x).mean(axis=1),torch.zeros(N,).to(self.device))
        for j in range(q):
            z = self.pdf.sample((N,)).float().to(self.device)
            xez = x*torch.exp(z)
            fxez = payoff(xez)
            fx = payoff(x.reshape(-1,self.dim))
            df += (fxez-fx).reshape(-1,1)
        return (self.iann(x)-1./q*df.reshape(-1,1))**2 # + self.iann(torch.zeros_like(x).to(self.device))**2
    
    def _icost_function_q_bdf2(self,n,q=1): # loss function for the automatic integration
        self.x = self._sampling(n)
        N = self.x.shape[0]
        df = torch.zeros((N,1)).to(self.device)
        for j in range(q):
            z = self.pdf.sample((N,)).float().to(self.device)
            xez = self.x*torch.exp(z)
            fxez = self.model_prev(xez)
            f0xez = self.model_prev1(xez)
            fx = self.model_prev(self.x.reshape(-1,self.dim))
            f0x = self.model_prev1(self.x.reshape(-1,self.dim))
            df += 2.0*(fxez - fx).reshape(-1,1)-(f0xez - f0x).reshape(-1,1)
        return (self.iann(self.x)-1/q*df.reshape(-1,1))**2 #+ self.iann(0.01*self.x)**2

    # def _icost
  
    def _cost_function_euler(self, x):
        v = self.model(x)
        v0 = self.model(torch.zeros_like(x).to(self.device))
        v_prev = self.model_prev(x)
        # v_prev0 = self.model_prev0(x)
        di = {}    # d/dx_i, i=1,...,d
        di_prev = {}
        dv = torch.autograd.grad(v,
                                 x,
                                 grad_outputs=torch.ones_like(v),
                                 create_graph=True)[0]
        
        dv_prev = torch.autograd.grad(v_prev,
                                      x,
                                      grad_outputs=torch.ones_like(v),
                                      create_graph=True)[0]
        penalty = torch.zeros_like(v)
        L = torch.zeros_like(v)
        f = torch.zeros_like(v)

        for i in range(self.dim):
            unity = 1./self.dim*torch.ones_like(dv[:,i].reshape(-1,1))
            di[str(i)] = dv[:,i].reshape(-1,1)
            target = torch.relu(torch.minimum(dv[:,i].reshape(-1,1),unity))
            di_prev[str(i)] = dv_prev[:,i].reshape(-1,1)
            penalty += (di[str(i)]-target)**2.0

        for i in range(self.dim):
            ki = np.exp(self.muj[i]+0.5*self.sigmaj[i][i]) - 1.0
            rho_sigma = 0.0
            for j in range(self.dim):
                L += 0.25 * self.sigma[i]*self.rho[i][j]*self.sigma[j]*x[:,i].reshape(-1,1)*x[:,j].reshape(-1,1) * di[str(i)] * di[str(j)]
                f += 0.5*self.rho[i][j] * self.sigma[j] * self.sigma[i] * x[:,i].reshape(-1,1)* di_prev[str(i)]
            f += (-self.r + 0.5*self.sigma[i]**2 + ki*self.lam) * x[:,i].reshape(-1,1)* di_prev[str(i)]
        L += 0.5 * self.r * v * v


        cost = 0.0
        vol = 1.0 # self.xmax**self.dim/x.shape[0]
        if self.gh_levels==0:
            integral_ann = self.iann(x)
            cost = 0.5*(v-v_prev)**2 + self.dt*L + self.dt*f*v - self.dt*integral_ann*v# + self._icost_function_q_euler(q=1)
            cost = cost + penalty/self.dim + 0.1*v0**2
        else:
            cost = 0.5*(v-v_prev)**2 + self.dt*L + self.dt*f*v - self.dt*self._integral_gh_euler().reshape(-1,1)*v
            
        return cost * vol



    def _cost_function_bdf2(self, x):

        v = self.model(x)
        #vec0 = torch.zeros((1,self.dims)).requires_grad_()to(self.device))
        v0 = self.model(torch.zeros_like(x).to(self.device))
        #v1 = self.model(torch.ones_like(x).to(self.device))
        v_prev = self.model_prev(x)
        #v_prev0 = self.model_prev0(x)
        v_prev1 = self.model_prev1(x)
        di = {}    # d/dx_i, i=1,...,d
        di_prev = {}
        di_prev1 = {}
        dv = torch.autograd.grad(v,
                                 x,
                                 grad_outputs=torch.ones_like(v),
                                 create_graph=True)[0]
        
        dv_prev = torch.autograd.grad(v_prev,
                                      x,
                                      grad_outputs=torch.ones_like(v),
                                      create_graph=True)[0]

        dv_prev1 = torch.autograd.grad(v_prev1,
                                       x,
                                       grad_outputs=torch.ones_like(v),
                                       create_graph=True)[0]
        L = torch.zeros_like(v)
        f = torch.zeros_like(v)
        penalty = torch.zeros_like(v)

        for i in range(self.dim):
            unity = 1./self.dim*torch.ones_like(dv[:,i].reshape(-1,1))
            di[str(i)] = dv[:,i].reshape(-1,1)
            target = torch.relu(torch.minimum(dv[:,i].reshape(-1,1),unity))
            penalty += (di[str(i)]-target)**2.0
            di_prev[str(i)] = dv_prev[:,i].reshape(-1,1)
            di_prev1[str(i)] = dv_prev1[:,i].reshape(-1,1)
                

        for i in range(self.dim):
            ki = np.exp(self.muj[i]+0.5*self.sigmaj[i][i]) - 1.0
            rho_sigma = 0.0
            for j in range(self.dim):
                L += 0.25 * self.sigma[i]*self.rho[i][j]*self.sigma[j]*x[:,i].reshape(-1,1)*x[:,j].reshape(-1,1) * di[str(i)] * di[str(j)]
                f += 0.5*self.rho[i][j] * self.sigma[j] * self.sigma[i] * x[:,i].reshape(-1,1)*  (2.0*di_prev[str(i)]-di_prev1[str(i)])
            f += (-self.r + 0.5*self.sigma[i]**2  + ki*self.lam) * x[:,i].reshape(-1,1)* (2.0*di_prev[str(i)]-di_prev1[str(i)])
        L += 0.5 * self.r * v * v


        cost = 0.0
        vol = 1.0 # *self.xmax**self.dim/x.shape[0]
        if self.gh_levels > 0.0: # Gauss-Hermite
            integral0 = self._integral_gh_bdf2(which='prev').reshape(-1,1)
            integral1 = self._integral_gh_bdf2(which='prev1').reshape(-1,1)
            icontr = 2*integral0-integral1
            cost = 0.5*(v-4.0/3.0*v_prev + 1.0/3.0*v_prev1)**2 + 2.0/3.0 *(self.dt*L + self.dt*f*v - self.dt*icontr*v)
            #cost = cost*vol + v0**2
        else: # Automatic integration using ANN
            integral_ann = self.iann(x)
            # integral_ann0 = self.iann(torch.zeros_like(x).to(self.device))
            #integral_ann0 = self.iann(0.075*x)
            cost = 0.5*(v-4.0/3.0*v_prev + 1.0/3.0*v_prev1)**2 + 2.0/3.0 *(self.dt*L + self.dt*f*v - self.dt*integral_ann*v)
            cost = cost  + penalty/self.dim + 0.1*v0**2
        return cost*vol
    
    def _cost_function_init(self,x):
        v = self.model(x)
        # fit a Gaussian distribution of mean 1 and of std 0.25
        def fact(x):
            w = 1.0/self.dim
            sval = 0.0
            for i in range(self.dim):
                sval += w * torch.abs(x[:,i].reshape(-1,1))
            return sval
        f = fact(x)
        
        f_init = torch.where(f > torch.ones(x.shape).to(self.device), 0.05*torch.exp(-0.5*((f-1.0)/0.25*3)**2) , 0.05*torch.exp(-0.5*((f-1.0)/0.25*4)**2))
        cost_init = (v-f_init)**2 
        return cost_init

    def _preview(self,selfdate_str,time_step,prefix=''):
        def model_eval(selfdate_str, x, time_step):
            models_dir = './models'
            date_dir = models_dir + '/' + selfdate_str
            model_ts = torch.load(date_dir+'/'+prefix+'model_'+str(time_step))
            iann_ts = None
            if self.gh_levels == 0:
                iann_ts = torch.load(date_dir+'/'+prefix+'iann_'+str(time_step))
            n = len(x)
            v = np.zeros(n)
            a = np.zeros(n)
            for i, it in enumerate(x):
                tmp0 = (it,)*self.dim
                #         tmp0 = (it, it+0.2,)
                v[i] = model_ts(torch.tensor(tmp0, dtype=torch.float32).to(self.device))
                if self.gh_levels == 0:
                    a[i] = iann_ts(torch.tensor(tmp0, dtype=torch.float32).to(self.device))
            return v,a
        
        def payoff():
            x = np.linspace(0,3,1024)
            y = np.zeros((1024,))
            p = np.maximum(y,x-1)
            return p
        x = np.linspace(0,3,1024)
        p = payoff()
        yd,ad = model_eval(selfdate_str, x, time_step)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.grid()
        plt.plot(x,yd-p,lw=1.5)
        models_dir = './models'
        date_dir = models_dir + '/' + selfdate_str
        plt.savefig(date_dir+'/' +prefix +str(time_step)+'_fig.png')
        plt.close()
        if self.gh_levels==0:
            plt.figure()
            plt.grid()
            plt.plot(x,ad,lw=1.5)
            plt.savefig(date_dir+'/'+prefix + str(time_step)+'_iann.png')
            plt.close()
        
    def _integral_gh_euler(self):
        import numpy as np
        import Tasmanian
        x = self.x
        levels = self.gh_levels
        Sigma = np.array(self.sigmaj)
        S,L,_ = np.linalg.svd(Sigma)
        sqrtL = np.diag(L**0.5)
        A = S@sqrtL
        grid = Tasmanian.SparseGrid()
        grid.makeGlobalGrid(iDimension=self.dim,
                            iOutputs=1,
                            iDepth=levels,
                            sType='level',
                            sRule='gauss-hermite-odd',
                            fAlpha=0.,
                            fBeta=0.)
        points = grid.getPoints()
        z = np.sqrt(2)*(A@points.T).T
        z = torch.tensor(z.reshape(-1,1,self.dim), dtype=torch.float32).requires_grad_().to(self.device)
        lenz = z.shape[0]
        x = x.reshape(1,-1,self.dim)
        xez = torch.tensor(x*torch.exp(z), dtype=torch.float32).requires_grad_().to(self.device)
        xez = xez.reshape(-1,self.dim)
        fxez = self.model_prev(xez).reshape(lenz,-1, 1)
        fx = self.model_prev(x.reshape(-1,self.dim)).reshape(-1,1)  
        w = grid.getQuadratureWeights()
        w = torch.tensor(w.reshape(-1,1,1), dtype=torch.float32).requires_grad_().to(self.device)
        integral = (fxez*w).sum(axis=0)
        return (1./np.sqrt(np.pi)**self.dim)*integral - fx

    # def _integral_gh_bdf2(self,which='prev'): 
    #     import numpy as np
    #     import Tasmanian
    #     x = self.x
    #     levels = self.gh_levels
    #     Sigma = np.array(self.sigmaj)
    #     S,L,_ = np.linalg.svd(Sigma)
    #     sqrtL = np.diag(L**0.5)
    #     A = S@sqrtL
    #     grid = Tasmanian.SparseGrid()
    #     grid.makeGlobalGrid(iDimension=self.dim,
    #                         iOutputs=1,
    #                         iDepth=levels,
    #                         sType='level',
    #                         sRule='gauss-hermite-odd',
    #                         fAlpha=0.,
    #                         fBeta=0.)
    #     points = grid.getPoints()
    #     z = np.sqrt(2)*(A@points.T).T
    #     z = torch.tensor(z.reshape(-1,1,self.dim), dtype=torch.float32).requires_grad_().to(self.device)
    #     lenz = z.shape[0]
    #     x = x.reshape(1,-1,self.dim)
    #     xez = torch.tensor(x*torch.exp(z), dtype=torch.float32).requires_grad_().to(self.device)
    #     xez = xez.reshape(-1,self.dim)
    #     fxez = None
    #     fx = None
    #     if which=='prev':
    #         fxez = self.model_prev(xez).reshape(lenz,-1, 1)
    #         fx = self.model_prev(x.reshape(-1,self.dim)).reshape(-1,1)
    #     elif which=='prev1':
    #         fxez = self.model_prev1(xez).reshape(lenz,-1, 1)
    #         fx = self.model_prev1(x.reshape(-1,self.dim)).reshape(-1,1)            
    #     w = grid.getQuadratureWeights()
    #     w = torch.tensor(w.reshape(-1,1,1).requires_grad_().to(self.device)
    #     integral = (fxez*w).sum(axis=0)
    #     return (1./np.sqrt(np.pi)**self.dim)*integral - fx
        
    def loss(self, n, initial=True):
        if initial == True:
            fc = 2*3.5*np.random.rand()
            self.x = self._sampling_init(n, xmax=3.0)
            cost0 = self._cost_function_init(self.x)
            return torch.mean(cost0)
        elif initial == False:
            self.x = self._sampling(n)
            # cost = None
            if self.scheme == 'euler':
                cost = self._cost_function_euler(self.x)
                return torch.mean(cost) 
            elif self.scheme == 'bdf2':
                cost = self._cost_function_bdf2(self.x)
                return torch.mean(cost) 

    def init_fit(self, model, n=1024, epoch=128, lr=3e-4):
        self.time_step = 0
        self.model = model.to(self.device)
        self.model.t = -1.0
        opt = torch.optim.Adam(self.model.parameters(), lr)
        for epoch in range(epoch):
            opt.zero_grad()
            loss = self.loss(n, initial=True)
            if epoch % 100 == 0:
                eloss="{:e}".format(loss)
                print(epoch, eloss)
            loss.backward()
            opt.step()
        self.model.t = 0.0
        self._save_init_model()
        self.time_step = 1

    def init_iann_fit(self, iann, n=1024, epoch=128, lr=3e-4):
        opt = torch.optim.Adam(self.iann.parameters(), lr)
        for epoch in range(epoch):
            opt.zero_grad()
            x = self._sampling_init(n, xmin=0.0, xmax=3.0)
            loss = self._icost_function_q_init(x,q=self.dim).to(self.device).mean()
            if epoch % 100 == 0:
                eloss="{:e}".format(loss)
                print(epoch, eloss)
            loss.backward()
            opt.step()
        self._save_init_iann()
            
            

    # start from the 1st
    def init_load_euler(self, model, resume_time=0):
        self.model = model.to(self.device)
        self.time_step = resume_time + 1


     # load the first two initial models, start from the 3rd
    def init_load_bdf2(self, model_prev, model, resume_time=0):
        self.model = model.to(self.device)
        self.model.t = (resume_time + 2)*self.dt
        self.model_prev = model_prev.to(self.device)
        self.model_prev.t = (resume_time+1)* self.dt
        self.time_step = resume_time + 3

                        


    def fit(self, n=1024, epoch=128, lr=3e-4, initial=False):
        from itertools import chain
        if self.scheme == 'bdf2':
            self.model_prev1 = copy.deepcopy(self.model_prev)
        self.model_prev = copy.deepcopy(self.model)
        # self.model_prev1 = None
        # self.model_prev0 = copy.deepcopy(self.model)
        self.model.t = self.time_step*self.dt
        opt = None
        iopt = None
        params = []
        params.append(chain(self.model.parameters()))
        #if self.gh_levels == 0:
        params_integral = []
        params_integral.append(chain(self.iann.parameters()))
        opt = torch.optim.Adam(chain(*params), lr) # Adam optimization algorithm
        opt_integral = torch.optim.Adam(chain(*params_integral), lr) # Adam optimization algorithm

        for epoch in range(epoch):
            opt_integral.zero_grad()
            loss_integral = None
            if self.scheme == 'bdf2':
                loss_integral = torch.mean(self._icost_function_q_bdf2(n, q=1))
            else:
                loss_integral = torch.mean(self._icost_function_q_euler(n, q=1))
            loss_integral.backward()
            opt_integral.step()
        #v0 = 0.
        #v1 = 0.
        for epoch in range(epoch):
            opt.zero_grad()
            loss = self.loss(n, initial)
            # print(loss)
            #x = torch.tensor([[1.0]*self.dim],dtype=torch.float32).to(self.device)
            #v1 = self.model(x).detach().cpu()
            #v1 = v1[[0]]
            #ratio = np.abs(v1-v0)/v1
            #print(ratio)
            if epoch % 100 == 0:
                # print(ratio)
                eloss="{:e}".format(loss)
                print(epoch, eloss)
            #v0 = v1
            #if ratio < 1e-5:
            #    break
            loss.backward()
            opt.step()
            if self.time_step == 1 and epoch%1000==0:
                prefix = '--' + str(epoch) + '--'
                self._save_cur_model(prefix=prefix)
            
        if self.time_step == 0:
            self._save_init_model()
        else:
            self._save_cur_model()
        self.time_step += 1

    def _save_cur_model(self,prefix=''):
        models_dir = './models'
        date_dir = models_dir + '/' + self.date_str
        self.date_dir = date_dir
        models_dir_exist = os.path.exists(models_dir)
        if not models_dir_exist:
            os.makedirs(models_dir)
        date_dir_exist = os.path.exists(date_dir)
        if not date_dir_exist:
            os.makedirs(date_dir)
            with open(date_dir + '/model.txt', 'w') as f:
                txt = 'Merton Model\n'
                txt += '-----------------------\n'
                txt += 'number of assets : ' + str(self.dim) + '\n'
                txt += 'Σ = ' + str(self.sigma) + '\n'
                txt += 'ρ = ' + str(self.rho) + '\n'
                txt += 'r = ' + str(self.r) + '\n'
                txt += 'λ = ' + str(self.lam) + '\n'
                txt += 'μj = ' + str(self.muj) + '\n'
                txt += 'Σj = ' + str(self.sigmaj) + '\n'
                txt += 'xmax = ' + str(self.xmax) + '\n'
                txt += 'T = ' + str(self.tmax) + '\n'
                txt += 'dt = ' + str(self.dt) + '\n'
                txt += 'gh_levels = ' + str(self.gh_levels) + '\n'
                f.write(txt)
        filename = date_dir + '/'+prefix+'model_' +  str(self.time_step)
        ifilename = date_dir + '/'+prefix+'iann_' +  str(self.time_step)
        torch.save(self.model, filename)
        if self.gh_levels==0:
            torch.save(self.iann, ifilename)
        self._preview(self.date_str,self.time_step,prefix=prefix)

    def _save_init_model(self,prefix=''):
        init_models_dir = './init'
        init_models_dir_exist = os.path.exists(init_models_dir)
        if not init_models_dir_exist:
            os.makedirs(init_models_dir)
        filename = init_models_dir + '/'+prefix+'dim_mod0'+str(self.dim)
        torch.save(self.model, filename)

    def _save_init_iann(self,prefix=''):
        init_models_dir = './init'
        init_models_dir_exist = os.path.exists(init_models_dir)
        if not init_models_dir_exist:
            os.makedirs(init_models_dir)
        filename = init_models_dir + '/'+prefix+'dim_iann0'+str(self.dim)
        torch.save(self.iann, filename)

    def _trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

