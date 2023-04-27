# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 04:04:03 2023

@author: iPM-Lab
"""
import torch
import torch.nn as nn
import numpy as np
import hamiltorch
import pickle as pk
import matplotlib.pyplot as plt

#%% BNN architecture
lrelu = nn.LeakyReLU(0.1)
sig = nn.Sigmoid()

class Net(nn.Module):

    def __init__(self, layer_sizes, loss = 'multi_class', bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.loss = loss
        self.bias = bias
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1],bias=True)
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2],bias = self.bias)
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3],bias = self.bias)

    def forward(self, x):
        x = self.l1(x)
        x = lrelu(x)
        x = self.l2(x)
        x = lrelu(x)
        x = self.l3(x)
        x = sig(x)
        
        return x


#%% Load trained HMC-BNNs
net0 = Net([6,40,40,1])
net0.load_state_dict(torch.load('Modelo_0.pt'))
net0.eval()

net1 = Net([6,40,40,1])
net1.load_state_dict(torch.load('Modelo_1.pt'))
net1.eval()

net2 = Net([6,40,40,1])
net2.load_state_dict(torch.load('Modelo_2.pt'))
net2.eval()

net3 = Net([6,40,40,1])
net3.load_state_dict(torch.load('Modelo_3.pt'))
net3.eval()

net4 = Net([6,40,40,1])
net4.load_state_dict(torch.load('Modelo_4.pt'))
net4.eval()

net5 = Net([6,40,40,1])
net5.load_state_dict(torch.load('Modelo_5.pt'))
net5.eval()

with open('Modelo0.dat','rb') as f:
    y_val0,params_hmc0,tau_list0 = pk.load(f)
    
with open('Modelo1.dat','rb') as f:
    y_val1,params_hmc1,tau_list1 = pk.load(f)

with open('Modelo2.dat','rb') as f:
    y_val2,params_hmc2,tau_list2 = pk.load(f)

with open('Modelo3.dat','rb') as f:
    y_val3,params_hmc3,tau_list3 = pk.load(f)

with open('Modelo4.dat','rb') as f:
    y_val4,params_hmc4,tau_list4 = pk.load(f)

with open('Modelo5.dat','rb') as f:
    y_val5,params_hmc5,tau_list5 = pk.load(f)


#%% Prediction function
def prediccion(l_lambda,fm,Em,tw,hw,lw,m_type,P=0, As=0):
    tau_out = 100
    
    ejemplo = [(hw/tw)/30, hw/lw, P/(tw*fm*lw), l_lambda/10 ,  As, m_type]
    x_val = torch.Tensor([ejemplo]).float()
    
    y_predicted0, log_prob_list0 = hamiltorch.predict_model(net0, x=x_val,y=y_val0, model_loss='regression', samples=params_hmc0[:],
    tau_out=tau_out, tau_list=tau_list0)
    y_predicted0 = y_predicted0.cpu().numpy().squeeze().T
    
    y_predicted1, log_prob_list1 = hamiltorch.predict_model(net1, x=x_val,y=y_val1, model_loss='regression', samples=params_hmc1[:],
    tau_out=tau_out, tau_list=tau_list1)
    y_predicted1 = y_predicted1.cpu().numpy().squeeze().T
    
    y_predicted2, log_prob_list2 = hamiltorch.predict_model(net2, x=x_val,y=y_val2, model_loss='regression', samples=params_hmc2[:],
    tau_out=tau_out, tau_list=tau_list2)
    y_predicted2 = y_predicted2.cpu().numpy().squeeze().T
    
    y_predicted3, log_prob_list3 = hamiltorch.predict_model(net3, x=x_val,y=y_val3, model_loss='regression', samples=params_hmc3[:],
    tau_out=tau_out, tau_list=tau_list3)
    y_predicted3 = y_predicted3.cpu().numpy().squeeze().T
    
    y_predicted4, log_prob_list4 = hamiltorch.predict_model(net4, x=x_val,y=y_val4, model_loss='regression', samples=params_hmc4[:],
    tau_out=tau_out, tau_list=tau_list4)
    y_predicted4 = y_predicted4.cpu().numpy().squeeze().T
    
    y_predicted5, log_prob_list5 = hamiltorch.predict_model(net5, x=x_val,y=y_val5, model_loss='regression', samples=params_hmc5[:],
    tau_out=tau_out, tau_list=tau_list5)
    y_predicted5 = y_predicted5.cpu().numpy().squeeze().T
    
    # np.max corrects the values in order to be able to use the uniaxial
    # material hysteretic from opensees
    dy = y_predicted0*hw/20
    dc = np.max([y_predicted1*hw/20,dy*1.01],axis=0)
    Vy = y_predicted2*1200
    Vc = np.max([y_predicted3*1200,Vy*1.1],axis=0)
    du = np.max([y_predicted4*hw/20,dc*1.01],axis=0)
    Vu = y_predicted5*1200
    
    return dy, dc, du, Vy, Vc, Vu

#%% Example of a prediction
''' 
l_lambda: lambda parameter (see eq 1 from Barros et al.)
fm: masonry characteristic compressive strength, in MPa
Em: masonry elastic modulus, in MPa
tw, hw, lw: thickness, height and length of the masonry wall, in mm
m_type:
    Autoclaved aereated concrete units = 1/6
    Limestone units = 2/6
    Hollow clay units = 3/6
    Solid clay units = 4/6
    Hollow concrete unit = 5/6
    Solid concrete unit = 1
P: axial load, in N
As: ratio of area of equivalent strut considering openings over 
    the area of equivalent strut
'''
l_lambda,fm,Em,tw,hw,lw,m_type,P,As = 1.535,10,9000,100,3000,3000,5/6,0,0
dy, dc, du, Vy, Vc, Vu = prediccion(l_lambda,fm,Em,tw,hw,lw,m_type,P,As)


#%% Results of a prediction
plt.figure()
plt.subplot(2,3,1); plt.hist(dy,label=f'$\mu = ${np.round(np.mean(dy),2):.2f}'); plt.title(r'$d_y$'); plt.legend()
plt.subplot(2,3,2); plt.hist(dc,label=f'$\mu = ${np.round(np.mean(dc),2):.2f}'); plt.title(r'$d_c$'); plt.legend()
plt.subplot(2,3,4); plt.hist(Vy,label=f'$\mu = ${np.round(np.mean(Vy),2):.2f}'); plt.title(r'$V_y$'); plt.legend()
plt.subplot(2,3,5); plt.hist(Vc,label=f'$\mu = ${np.round(np.mean(Vc),2):.2f}'); plt.title(r'$V_c$'); plt.legend()
plt.subplot(2,3,3); plt.hist(du,label=f'$\mu = ${np.round(np.mean(du),2):.2f}'); plt.title(r'$d_u$'); plt.legend()
plt.subplot(2,3,6); plt.hist(Vu,label=f'$\mu = ${np.round(np.mean(Vu),2):.2f}'); plt.title(r'$V_u$'); plt.legend()
plt.tight_layout()



