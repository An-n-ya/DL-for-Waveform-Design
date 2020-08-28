#!/usr/bin/env python
# coding: utf-8



import time
import torch
from torch import nn, optim
import torch.nn.functional as F
#import d2lzh_pytorch as d2l
import cmath
import numpy as np
from numpy import mat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Residual(nn.Module):
    def __init__(self, J, ML, first_block = False, last_block = False):
        super(Residual, self).__init__()
        self.layer1 = nn.Linear(J,J)
        self.layer2 = nn.Linear(J,J)
        if first_block:
            self.layer1 = nn.Linear(ML,J)
            self.layer3 = nn.Linear(ML,J)
        elif last_block:
            self.layer3 = nn.Linear(J,ML)
            self.layer2 = nn.Linear(J,ML)
        else:
            self.layer3 = None
        #self.bn1 = nn.InstanceNorm1d(J)
        #self.bn2 = nn.InstanceNorm1d(J)
    def forward(self, X):
        #Y = F.relu(self.bn1(self.layer1(X)))
        #Y = self.layer2(self.bn2(Y))
        Y = F.relu(self.layer1(X))
        Y = self.layer2(Y)
        if self.layer3:
            X = self.layer3(X)
        return F.relu(Y+X)


J, M, L = 128, 10, 32
ML = M * L
net = nn.Sequential(
        Residual(J, ML, first_block=True),
        Residual(J, ML),
        Residual(J, ML),
        Residual(J, ML),
        Residual(J, ML, last_block=True),
        )

def at(theta,M):
    #return torch.exp(-1j * 2 * torch.tensor(np.pi) * torch.sin(torch.deg2rad(theta)) * torch.arange(M))
    return np.exp(-1j * 2 * np.pi * np.sin(np.deg2rad(theta)) * mat(list(range(M))))
def design(theta):
    #one = torch.ones(theta.shape[0])
    one = np.ones(theta.shape[0])
    one[theta > 10] = 0
    return one
def loss_func(phi,L,device):
    mat_phi = phi.reshape(-1,L)
    M = mat_phi.shape[0]
    #P = torch.zeros(100)
    P = list()
    for idx, theta in enumerate(torch.arange(-10,10,0.2)):
        Re_at = np.real(at(theta, M))
        Im_at = np.imag(at(theta, M))
        Re_at = torch.from_numpy(Re_at).to(device)
        Im_at = torch.from_numpy(Im_at).to(device)
        #Re_at = at(theta, M).real.to(torch.float64)
        #Im_at = at(theta, M).imag.to(torch.float64)
        Re_y = torch.matmul(Re_at, torch.cos(mat_phi).to(torch.float64)) - torch.matmul(Im_at, torch.sin(mat_phi).to(torch.float64))
        Im_y = torch.matmul(Re_at, torch.sin(mat_phi).to(torch.float64)) + torch.matmul(Im_at, torch.cos(mat_phi).to(torch.float64))
        #P[idx] = (torch.pow(Re_y,2).sum() + torch.pow(Im_y,2).sum())
        P.append(torch.pow(Re_y,2).sum() + torch.pow(Im_y,2).sum())
    des = design(np.arange(-10,10,0.2))
    des = torch.from_numpy(des).to(device)
    P = torch.tensor(P).to(device)
    u_opt = (P * des).sum() / torch.pow(des, 2).sum()
    F1 = torch.pow((u_opt * des) - P, 2).sum() / des.shape[0]
    F2 = 0
    j = 0
    for theta in np.concatenate((np.arange(-90,-10,0.2),np.arange(10,90,0.2))):
        Re_at = np.real(at(theta, M))
        Im_at = np.imag(at(theta, M))
        Re_at = torch.from_numpy(Re_at).to(device)
        Im_at = torch.from_numpy(Im_at).to(device)
        Re_y = torch.matmul(Re_at, torch.cos(mat_phi).to(torch.float64)) - torch.matmul(Im_at, torch.sin(mat_phi).to(torch.float64))
        Im_y = torch.matmul(Re_at, torch.sin(mat_phi).to(torch.float64)) + torch.matmul(Im_at, torch.cos(mat_phi).to(torch.float64))
        F2 += torch.pow(Re_y,2).sum() + torch.pow(Im_y,2).sum()
        j += 1
    F2 = F2 / j
    return 10 * F2 + F1

def train(net, X, optimizer, device, num_epochs, L):
    net = net.to(device)
    print("training on", device)
    train_l_sum = 0
    for epoch in range(num_epochs):
        X = X.to(device)
        Y = net(X)
        l = loss_func(Y * 2 * 3.14159, L,device)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum = l.cpu().item()
        print('epoch %d, loss %.4f'%(epoch+1, train_l_sum))
    


# # Training Model


X = torch.rand(ML)

lr, num_epochs = 0.00005, 60
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
train(net, X, optimizer, device, num_epochs, L)

result = net(X.to(device))
mat_result = result.reshape(-1,L)
mat_result.to('cpu')

P = list()
for idx, theta in enumerate(torch.arange(-90,90,0.2)):
        Re_at = np.real(at(theta, M))
        Im_at = np.imag(at(theta, M))
        Re_at = torch.from_numpy(Re_at).to('cpu')
        Im_at = torch.from_numpy(Im_at).to('cpu')
        #Re_at = at(theta, M).real.to(torch.float64)
        #Im_at = at(theta, M).imag.to(torch.float64)
        Re_y = torch.matmul(Re_at, torch.cos(mat_result).to(torch.float64)) - torch.matmul(Im_at, torch.sin(mat_result).to(torch.float64))
        Im_y = torch.matmul(Re_at, torch.sin(mat_result).to(torch.float64)) + torch.matmul(Im_at, torch.cos(mat_result).to(torch.float64))
        #P[idx] = (torch.pow(Re_y,2).sum() + torch.pow(Im_y,2).sum())
        P.append(torch.pow(Re_y,2).sum() + torch.pow(Im_y,2).sum())
P = torch.tensor(P).to('cpu')

np_P = P.numpy()
from matplotlib import pyplot as plt
plt.plot(np.arange(-90,90,0.2),np_P)

#X = torch.rand(20)
#device = torch.device('cpu')
#loss_func(X, 5, device)



