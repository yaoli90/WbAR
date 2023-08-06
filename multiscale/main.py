import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import numpy as np
import time
from scipy.stats import qmc
import scipy.io

def training_data_latin_hypercube(X, N_inner=1e3):

    # Latin Hypercube sampling for collocation points
    lb = np.min(X)
    ub = np.max(X)
    sampler = qmc.LatinHypercube(d=1)
    x_train = lb + (ub-lb)*sampler.random(n=int(N_inner))

    x_boundary = np.array([lb, ub]).reshape(2,1)
    u_boundary = np.array([0, 0]).reshape(2,1)

    return x_train, x_boundary, u_boundary

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

        x.requires_grad = True

        epsilon = 1/4
        ax = 0.5*torch.sin(2*np.pi*x/epsilon) + torch.sin(x) + 2

        u = self.forward(x)

        u_x = autograd.grad(u,x,torch.ones(x.shape).to(self.device), retain_graph=True, create_graph=True)[0]
        u_xx = autograd.grad(u_x*ax, x, torch.ones(x.shape).to(self.device), retain_graph = True, create_graph = True)[0]

        f = u_xx + torch.sin(x)
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
        return 200*loss_u + loss_f

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
