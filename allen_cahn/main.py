import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import numpy as np
import time
from scipy.stats import qmc
import scipy.io

def training_data_latin_hypercube(X, T, U_gt, N_boundary=400, N_inner=1e3, lb=None, ub=None):

    # boundary conditions
    x_t_left = np.hstack((X[0,:][:,None], T[0,:][:,None]))
    u_left = U_gt[:,0][:,None]

    x_t_bottom = np.hstack((X[:,0][:,None], T[:,0][:,None]))
    u_bottom = U_gt[-1,:][:,None]

    x_t_top = np.hstack((X[:,-1][:,None], T[:,0][:,None]))
    u_top = U_gt[0,:][:,None]

    x_t_boundary = np.vstack([x_t_left, x_t_bottom, x_t_top])
    u_boundary = np.vstack([u_left, u_bottom, u_top])

    # choose random N_boundary points for training
    idx = np.random.choice(x_t_boundary.shape[0], N_boundary, replace=False)
    x_t_boundary = x_t_boundary[idx, :]
    u_boundary = u_boundary[idx,:]

    # Latin Hypercube sampling for collocation points
    x_t_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    if lb is None:
        lb = x_t_test[0,:]  # [-1. 0.]
    if ub is None:
        ub = x_t_test[-1,:] # [1.  0.99]
    sampler = qmc.LatinHypercube(d=2)
    x_t_inner = lb + (ub-lb)*sampler.random(n=int(N_inner))
    # append training points to collocation points
    x_t_train = np.vstack((x_t_inner, x_t_boundary))
    return x_t_train, x_t_boundary, u_boundary

class sequential_model(nn.Module):

    def __init__(self, layers, device):
        super().__init__()
        self.layers = layers
        self.device = device
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0

        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    '''
    Forward Propagation
        input: x := [x, t]
        output: u(x,theta) '''
    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        for i in range(len(self.layers)-2):
            z = self.linears[i](x)
            x = self.activation(z)

        output = self.linears[-1](x)
        return output

    '''
    Model Residual
        input: x := [x, t]
        output: r(x,theta) '''
    def function(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        D = 0.0001

        g = x
        g.requires_grad = True

        u = self.forward(g)
        u_xt = autograd.grad(u,g,torch.ones([x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_xxtt = autograd.grad(u_xt,g,torch.ones(x.shape).to(self.device), create_graph=True)[0]

        u_t = u_xt[:,[1]]
        u_xx = u_xxtt[:,[0]]

        f = u_t - 5*(self.forward(g)-torch.pow(self.forward(g),3)) - D*u_xx
        return f

    def loss_BC(self, x_boundary, u_boundary):
        return self.loss_function(self.forward(x_boundary), u_boundary)

    def loss_PDE(self, f):
        loss_f = self.loss_function(f, torch.zeros(f.shape).to(self.device))
        return loss_f

    def loss(self, x_boundary, u_boundary, x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)
        if torch.is_tensor(x_boundary) != True:
            x_boundary = torch.from_numpy(x_boundary).float().to(self.device)
        if torch.is_tensor(u_boundary) != True:
            u_boundary = torch.from_numpy(u_boundary).float().to(self.device)

        f = self.function(x)

        loss_u = self.loss_BC(x_boundary, u_boundary)
        loss_f = self.loss_PDE(f)
        return loss_u + loss_f

    def train_model_adam(self, optimizer, x_boundary, u_boundary, x_train, n_epoch):
        while self.iter < n_epoch:
            optimizer.zero_grad()
            loss = self.loss(x_boundary, u_boundary, x_train)
            loss.backward()
            self.iter += 1
            if self.iter % 1000 == 0:
                print(self.iter, loss)
            optimizer.step()

    '''
    Test Model
        input: x, u
        output: rmse, u_pred '''
    def test(self, x, u):
        if torch.is_tensor(u) != True:
            u = torch.from_numpy(u).float().to(self.device)
        u_pred = self.forward(x)
        rmse = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)
        u_pred = u_pred.cpu().detach().numpy()
        return rmse, u_pred
