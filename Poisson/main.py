import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import numpy as np
import time
from scipy.stats import qmc
import scipy.io

def training_data_latin_hypercube(X, Y, U_gt, N_boundary=200, N_inner=1e3):

    '''Boundary Conditions'''

    #Initial Condition -1 =< x =<1 and t = 0
    x_y_left = np.hstack((X[:,0][:,None], Y[:,0][:,None])) #L1
    u_left = U_gt[:,0][:,None]

    x_y_right = np.hstack((X[:,-1][:,None], Y[:,-1][:,None])) #L1
    u_right = U_gt[:,-1][:,None]

    #Boundary Condition x = -1 and 0 =< t =<1
    x_y_bottom = np.hstack((X[-1,:][:,None], Y[-1,:][:,None])) #L2
    u_bottom = U_gt[-1,:][:,None]

    #Boundary Condition x = 1 and 0 =< t =<1
    x_y_top = np.hstack((X[0,:][:,None], Y[0,:][:,None])) #L3
    u_top = U_gt[0,:][:,None]

    x_y_boundary = np.vstack([x_y_left, x_y_right, x_y_bottom, x_y_top]) # X_u_train [456,2] (456 = 256(L1)+100(L2)+100(L3))
    u_boundary = np.vstack([u_left, u_right, u_bottom, u_top])         #corresponding u [456x1]

    #choose random N_u points for training
    idx = np.random.choice(x_y_boundary.shape[0], N_boundary, replace=False)
    x_y_boundary = x_y_boundary[idx, :] #choose indices from  set 'idx' (x,t)
    u_boundary = u_boundary[idx,:]      #choose corresponding u

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    x_y_test = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
    lb = x_y_test[0,:]  # [-1. 0.]
    ub = x_y_test[-1,:] # [1.  0.99]
    sampler = qmc.LatinHypercube(d=2)
    x_y_inner = lb + (ub-lb)*sampler.random(n=int(N_inner))
    #x_y_inner = lb + (ub-lb)*lhs(2, int(N_inner))
    x_y_train = np.vstack((x_y_inner, x_y_boundary)) # append training points to collocation points

    return x_y_train, x_y_boundary, u_boundary

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
        input: x := [x, y]
        output: r(x,theta) '''
    def function(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        g = x
        g.requires_grad = True

        u = self.forward(g)

        u_x_y = autograd.grad(u,g,torch.ones([x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_x = u_x_y[:,[0]]
        u_y = u_x_y[:,[1]]

        u_xx_xy = autograd.grad(u_x, g, torch.ones([x.shape[0], 1]).to(self.device), retain_graph = True, create_graph = True)[0]
        u_xx = u_xx_xy[:,[0]]

        u_yx_yy = autograd.grad(u_y, g, torch.ones([x.shape[0], 1]).to(self.device), retain_graph = True, create_graph = True)[0]
        u_yy = u_yx_yy[:,[1]]

        x_ = x[:,[0]]
        y_ = x[:,[1]]
        ex1 = torch.exp(-torch.square(10*x_-8))
        ey1 = torch.exp(-torch.square(10*y_-8))
        ex2 = torch.exp(-torch.square(10*x_+8))
        ey2 = torch.exp(-torch.square(10*y_+8))
        ex3 = torch.exp(-torch.square(10*x_))
        ey3 = torch.exp(-torch.square(10*y_))
        f = u_xx + u_yy \
            + 200*ex1 - 40000*torch.square(x_-.8)*ex1 -(200*ey1 - 40000*torch.square(y_-.8)*ey1)\
            + 200*ex2 - 40000*torch.square(x_+.8)*ex2 -(200*ey2 - 40000*torch.square(y_+.8)*ey2)\
            + 200*ex3 - 40000*torch.square(x_)*ex3 -(200*ey3 - 40000*torch.square(y_)*ey3)
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
