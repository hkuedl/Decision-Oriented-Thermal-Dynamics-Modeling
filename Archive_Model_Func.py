#%%
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import random
import torch
def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

set_seed(20)
import torch.nn as nn
import torch.optim as optim
from torchdiffeq1 import odeint
device = torch.device("cuda:2")
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,N_zone,lack_ratio):
    T_len = 8760*T_Fre
    if N_zone == 6:
        data_in = np.loadtxt("Archive_Data/6zone.csv",delimiter=",",skiprows=1,usecols=range(1,32))
        P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
        T_rad0 = data_in[:,7:13]/1000
        for i in range(N_zone):
            P0[:,i] = data_in[:,15+3*i]/1000
            T_in0[:,i] = data_in[:,13+3*i]
        T_o0 = data_in[:,0].reshape(-1,1)
        T_occ0 = data_in[:,1:7].copy()/1000
    elif N_zone == 10:
        data_in = np.loadtxt("Archive_Data/10zone.csv",delimiter=",",skiprows=1,usecols=range(1,52))
        P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
        T_rad0 = data_in[:,11:21]/1000
        for i in range(N_zone):
            P0[:,i] = data_in[:,23+3*i]/1000
            T_in0[:,i] = data_in[:,21+3*i]
        T_o0 = data_in[:,0].reshape(-1,1)
        T_occ0 = data_in[:,1:11].copy()/1000
    elif N_zone == 18:
        data_in = np.loadtxt("Archive_Data/18zone.csv",delimiter=",",skiprows=1,usecols=range(1,92))
        P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
        T_rad0 = data_in[:,19:37]/1000
        for i in range(N_zone):
            P0[:,i] = data_in[:,39+3*i]/1000 
            T_in0[:,i] = data_in[:,37+3*i]
        T_o0 = data_in[:,0].reshape(-1,1)
        T_occ0 = data_in[:,1:19].copy()/1000
    elif N_zone == 67:
        data_in = np.loadtxt("Archive_Data/67zone.csv",delimiter=",",skiprows=1,usecols=range(1,67*5+2))
        P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
        T_rad0 = data_in[:,(67+1):(2*67+1)]/1000
        for i in range(N_zone):
            P0[:,i] = data_in[:,(2*67+3)+3*i]/1000 
            T_in0[:,i] = data_in[:,(2*67+1)+3*i]
        T_o0 = data_in[:,0].reshape(-1,1)
        T_occ0 = data_in[:,1:(67+1)].copy()/1000
    elif N_zone == 90:
        data_in = np.loadtxt("Archive_Data/90zone.csv",delimiter=",",skiprows=1,usecols=range(1,90*5+2))
        P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
        T_rad0 = data_in[:,(90+1):(2*90+1)]/1000
        for i in range(N_zone):
            P0[:,i] = data_in[:,(2*90+3)+3*i]/1000 
            T_in0[:,i] = data_in[:,(2*90+1)+3*i]
        T_o0 = data_in[:,0].reshape(-1,1)
        T_occ0 = data_in[:,1:(90+1)].copy()/1000
    
    lack_list = random.sample(range(N_zone), int(N_zone*lack_ratio))
    P0[:,lack_list] = 0
    T_rad0[:,lack_list] = 0
    T_occ0[:,lack_list] = 0

    area = pd.read_excel('Archive_Data/Zone_area.xlsx', sheet_name=str(N_zone),header=None).values[:,0:1]
    TRUE_ratio = area/np.sum(area)
    P0n = np.sum(P0[:,:],1).reshape(-1,1)
    T_in = sum(TRUE_ratio[i]*T_in0[:,i] for i in range(N_zone)).reshape(-1,1)
    X_tr = np.hstack((T_o0[Train_s:Train_e,0:1],T_rad0[Train_s:Train_e,:],T_occ0[Train_s:Train_e,:]))
    P_tr = P0n[Train_s+1:Train_e+1,0:1]
    Y_ad_tr = np.zeros((Train_e-Train_s,3))
    Y_tr = T_in[Train_s:Train_e,0:1]
    X_te = np.hstack((T_o0[Train_s2:Train_e2,0:1],T_rad0[Train_s2:Train_e2,:],T_occ0[Train_s2:Train_e2,:]))
    P_te = P0n[Train_s2+1:Train_e2+1,0:1]
    Y_ad_te = np.zeros((Train_e2-Train_s2,3))
    Y_te = T_in[Train_s2:Train_e2,0:1]

    def reorder(X_tr, T_Fre):
        day = int((X_tr.shape[0]-1)/(T_Fre*24))
        for i in range(day-1):
            index = (i+1)*T_Fre*24+i
            X_tr = np.insert(X_tr, index, X_tr[index,:], axis=0)
        return X_tr
    X_tr1,P_tr1,Y_ad_tr1,Y_tr1,X_te1,P_te1,Y_ad_te1,Y_te1 = reorder(X_tr, T_Fre),reorder(P_tr, T_Fre),reorder(Y_ad_tr, T_Fre),reorder(Y_tr, T_Fre),reorder(X_te, T_Fre),reorder(P_te, T_Fre),reorder(Y_ad_te, T_Fre),reorder(Y_te, T_Fre)

    SS_X = MinMaxScaler().fit(X_tr1)
    SS_P = MinMaxScaler().fit(P_tr1)
    SS_Y = MinMaxScaler().fit(Y_tr1)
    SS_Y_ad = MinMaxScaler().fit(Y_ad_tr1)

    S_X_tr = SS_X.transform(X_tr1)
    S_P_tr = SS_P.transform(P_tr1)
    S_Y_tr = SS_Y.transform(Y_tr1)
    S_Y_ad_tr = SS_Y_ad.transform(Y_ad_tr1)
    S_X_te = SS_X.transform(X_te1)
    S_P_te = SS_P.transform(P_te1)
    S_Y_te = SS_Y.transform(Y_te1)
    S_Y_ad_te = SS_Y_ad.transform(Y_ad_te1)

    return SS_X,SS_P,SS_Y,SS_Y_ad,S_X_tr,S_P_tr,S_Y_tr,S_Y_ad_tr,S_X_te,S_P_te,S_Y_te,S_Y_ad_te,Y_tr1,Y_te1

class NeuralODE_ab(nn.Module):
    def __init__(self,n_zone,linear,y_phy):
        super(NeuralODE_ab, self).__init__()
        layers = 48
        self.a = nn.Parameter(torch.tensor(-0.2), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-0.5), requires_grad=True)
        self.y_phy = y_phy
        
        self.n_zone = n_zone
        self.input_num = 1 + self.n_zone*2
        if linear == 0:
            self.net = nn.Sequential(
                nn.Linear(self.input_num, layers),
                nn.ReLU(),
                nn.Linear(layers, layers),
                nn.ReLU(),
                nn.Linear(layers, layers),
                nn.ReLU(),
                nn.Linear(layers, 1),)
        elif linear == 1:
            self.net = nn.Sequential(
                nn.Linear(self.input_num, layers),
                nn.Linear(layers, layers),
                nn.Linear(layers, layers),
                nn.Linear(layers, 1),)

    def forward(self, t, z):
        y1, y2 = z[:,:,0:1].requires_grad_(True), z[:,:,1:2].requires_grad_(True)
        comp1 = self.a * y1 + self.b * y2
        x = []
        for i in range(self.input_num):
            x.append(z[:,:,(i+2):(i+3)].requires_grad_(True))
        comp2 = self.net(torch.cat([x[i] for i in range(self.input_num)], dim=-1))
        if self.y_phy == 1: 
            u_x = []
            if self.n_zone >= 19:
                sele_list = random.sample(range(self.n_zone), int(self.n_zone*0.1))
            else:
                sele_list = [i for i in range(self.n_zone)]
            u_x.append(torch.autograd.grad(comp2, x[0], grad_outputs=torch.ones_like(comp2),retain_graph=True,create_graph=True)[0])
            for i in sele_list:
                u_x.append(torch.autograd.grad(comp2, x[1+i], grad_outputs=torch.ones_like(comp2),retain_graph=True,create_graph=True)[0])
            for i in sele_list:
                u_x.append(torch.autograd.grad(comp2, x[1+self.n_zone+i], grad_outputs=torch.ones_like(comp2),retain_graph=True,create_graph=True)[0])
            
            u_y1 = torch.autograd.grad(comp1, y1, grad_outputs=torch.ones_like(comp1),retain_graph=True,create_graph=True)[0]
            u_y2 = torch.autograd.grad(comp1, y2, grad_outputs=torch.ones_like(comp1),retain_graph=True,create_graph=True)[0]
            
            relu = nn.ReLU()
            com_phy = [relu(u_y1),relu(u_y2)]
            com_phy_ori = [u_y1,u_y2]
            for i in range(len(u_x)):
                com_phy.append(relu(-1*u_x[i]))
                com_phy_ori.append(u_x[i])
            return comp1 + comp2, comp2, com_phy, com_phy_ori
        elif self.y_phy == 0:
            com_phy = [comp2,comp2]
            com_phy_ori = [comp2,comp2]
            for i in range(self.input_num):
                com_phy.append(comp2)
                com_phy_ori.append(comp2)
            return comp1 + comp2, comp2, com_phy, com_phy_ori

def Inteplo(T_y, step):
    m,n = T_y.shape
    NN_step = (n-1)*step + 1
    T_y_n = torch.zeros((m,NN_step))
    for ii in range(n):
        T_y_n[:,step*ii] = T_y[:,ii]
    for j1 in range(n-1):
        for j2 in range(1,step):
            T_y_n[:,step*j1+j2] = T_y[:,j1] + j2*(T_y[:,j1+1]-T_y[:,j1])/step
    return T_y_n

def getdata(batch_step_bei, T_Fre, S_Y_tr,S_X_tr,S_P_tr,S_Y_ad_tr, FF, n_zone):
    batch_size = int((S_P_tr.shape[0])/(24*T_Fre+1))  #how many days
    batch_days = [i for i in range(batch_size)]
    batch_num = 24*T_Fre+1
    batch_time = torch.arange(0., batch_num, FF).to(device)

    T_Y_tr = np.zeros((batch_size,len(batch_time)))
    for i in range(batch_size):
        for j in range(len(batch_time)):
            T_Y_tr[i,j] = S_Y_tr[i*(batch_num)+FF*j,0]
    T_Y_tr = torch.tensor(T_Y_tr).to(device)

    input_num = 1 + n_zone*2
    T_X_tr = np.zeros((input_num,batch_size,len(batch_time)))
    for i in range(input_num):
        for j in range(batch_size):
            for k in range(len(batch_time)):
                T_X_tr[i,j,k] = S_X_tr[j*(batch_num)+FF*k,i]
    T_X_tr = torch.tensor(T_X_tr).to(device)

    T_P_tr = np.zeros((1,batch_size,len(batch_time)))
    for i in range(batch_size):
        for j in range(len(batch_time)):
            T_P_tr[0,i,j] = S_P_tr[i*(batch_num)+FF*j,0]
    T_P_tr = torch.tensor(T_P_tr).to(device)

    N_adj = S_Y_ad_tr.shape[1]
    T_Y_ad_tr = np.zeros((N_adj,batch_size,len(batch_time)))
    for i in range(N_adj):
        for j in range(batch_size):
            for k in range(len(batch_time)):
                T_Y_ad_tr[i,j,k] = S_Y_ad_tr[j*(batch_num)+FF*k,i]
    T_Y_ad_tr = torch.tensor(T_Y_ad_tr).to(device)

    T_y_tr = torch.cat((T_P_tr,T_X_tr),dim = 0)

    batch_y0 = torch.zeros((len(batch_days),1,1)).to(device)  #(21,1,1)
    batch_y0[:,0,0] = T_Y_tr[batch_days,0]
    NN_step = (len(batch_time)-1)*batch_step_bei + 1
    batch_y11 = torch.zeros((NN_step,len(batch_days),1,T_y_tr.shape[0])).to(device)   #(191,21,1,5)
    if batch_step_bei == 1:
        for i in range(len(batch_days)):
            batch_y11[:,i,0,:] = torch.transpose(T_y_tr[:,i,:],0,1)
    else:
        for i in range(len(batch_days)):
            batch_ytt = Inteplo(T_y_tr[:,batch_days[i],:], batch_step_bei)
            batch_y11[:,i,0,:] = torch.transpose(batch_ytt,0,1)
    batch_y = torch.zeros((len(batch_time),len(batch_days),1,1)).to(device)  #(96,21,1,1)
    for i in range(len(batch_days)):
        batch_y[:,i,0,0] = T_Y_tr[batch_days[i],:]
    return T_Y_tr, batch_y0, batch_y11, batch_y

def verify_test(Y_te_pre,Y_te,len):
    T_len = Y_te.shape[0]
    T_len1 = int(Y_te.shape[0]*(len-1)/len)
    Err_tr = np.abs(Y_te_pre[:,0] - Y_te[:,0])
    Err_tr1 = math.sqrt(sum(Err_tr[i]**2/(T_len1) for i in range(T_len)))  #RMSE
    Err_tr2 = sum(Err_tr[i] for i in range(T_len))/(T_len1)  #MAE
    Err_tr3 = max(Err_tr)  #MAX
    Err_tr4 = r2_score(Y_te, Y_te_pre)  
    ERR2 = [Err_tr1,Err_tr2,Err_tr3,Err_tr4]
    Days = int(Y_te.shape[0]/len)
    Err_day = np.zeros((Days,1))
    for ii in range(Days):
        Err_day[ii,0] = math.sqrt(sum(Err_tr[i]**2/(len-1) for i in range(ii*len,(ii+1)*len)))  #RMSE
    return Err_tr,ERR2,Err_day

def Optimization(fre_opt, fre_cal, SS_Y, SS_P,nn_zone):
    c_dT = int(fre_cal*fre_opt)
    c_0_price = np.array(pd.read_excel('Archive_Data/Price_signal.xlsx'),dtype=float)
    c_0_tem = np.array(pd.read_excel('Archive_Data/temperature_limit.xlsx'))
    if nn_zone == 6:
        i_tem0,i_tem1 = 1,2
    elif nn_zone == 10:
        i_tem0,i_tem1 = 3,4
    elif nn_zone == 18:
        i_tem0,i_tem1 = 5,6
    elif nn_zone == 67:
        i_tem0,i_tem1 = 7,8
    elif nn_zone == 90:
        i_tem0,i_tem1 = 9,10

    c_time_p = int(96/fre_opt)
    c_time = int(96/fre_opt) + 1
    c_time_dd = (c_time-1)*c_dT + 1
    c_p_A,c_p_B,c_p_F = cp.Parameter(),cp.Parameter(),cp.Parameter(c_time_dd-1)  #after normalization
    c_v_tem,c_v_q = cp.Variable(c_time_dd),cp.Variable(c_time_p)  #after normalization
    c_v_tem_u,c_v_tem_l = cp.Variable(c_time, pos=True),cp.Variable(c_time, pos=True)   #after normalization: t/(tmax-tmin)

    c_upper_tem = np.zeros((c_time,1))   #without normalization
    c_lower_tem = np.zeros((c_time,1))
    c_upper_p = 60*np.ones((c_time_p,1))   #power, without normalization
    c_lower_p = 0*np.ones((c_time_p,1))
    c_price = np.zeros((c_time_p,1))
    for i in range(24):
        for j in range(int(c_time_p/24)):
            c_price[i*int(c_time_p/24)+j,0] = 0.001*c_0_price[i,1]
            c_lower_tem[i*int(c_time_p/24)+j,0] = c_0_tem[i,i_tem0]
            c_upper_tem[i*int(c_time_p/24)+j,0] = c_0_tem[i,i_tem1]
    c_lower_tem[-1,0],c_upper_tem[-1,0] = c_0_tem[0,i_tem0], c_0_tem[0,i_tem1]

    c_upper_tem = SS_Y.transform(c_upper_tem).ravel()
    c_lower_tem = SS_Y.transform(c_lower_tem).ravel()
    c_PI = 3.60   #COP
    c_upper_q = SS_P.transform(c_PI*c_upper_p).ravel()
    c_lower_q = SS_P.transform(c_PI*c_lower_p).ravel()

    c_Pmax,c_Pmin,c_tmax,c_tmin = SS_P.data_max_[0],SS_P.data_min_[0],SS_Y.data_max_[0],SS_Y.data_min_[0]
    c_tem_cost_u,c_tem_cost_l = 0.8*(c_tmax-c_tmin), 0.5*(c_tmax-c_tmin)   #true cost*（max-min)

    c_obj = sum(c_price[t,0]*(1/c_PI)*(0.25*fre_opt)*((c_Pmax-c_Pmin)*c_v_q[t]+c_Pmin) for t in range(c_time_p)) \
        +sum(c_tem_cost_u*c_v_tem_u[t] + c_tem_cost_l*c_v_tem_l[t] for t in range(c_time))

    c_obj1 = sum(c_price[t,0]*(1/c_PI)*(0.25*fre_opt)*((c_Pmax-c_Pmin)*c_v_q[t]+c_Pmin) for t in range(c_time_p))
    c_obj2 = sum(c_tem_cost_u*c_v_tem_u[t] + c_tem_cost_l*c_v_tem_l[t] for t in range(c_time))
    cons_tem1 = [c_v_tem[fre_cal*t] <= c_upper_tem[t]+c_v_tem_u[t] for t in range(c_time)]
    cons_tem2 = [c_v_tem[fre_cal*t] >= c_lower_tem[t]-c_v_tem_l[t] for t in range(c_time)]
    cons_q = [c_v_q <= c_upper_q, c_v_q >= c_lower_q]
    cons_model = [c_v_tem[t+1] == c_v_tem[t] + (1/fre_cal)*(c_p_A*c_v_tem[t] + c_p_B*c_v_q[int(t//c_dT)] + c_p_F[t]) for t in range(c_time_dd-1)]
    cons_ini = [c_v_tem[0] == c_lower_tem[0]]   #, c_v_tem[-1] == c_v_tem[0]]
    cons_tem11 = [c_v_tem_u[t] <= 0.5 for t in range(c_time)]
    cons_tem22 = [c_v_tem_l[t] <= 0.5 for t in range(c_time)]
    cons = cons_tem1 + cons_tem2 + cons_q + cons_model + cons_ini + cons_tem11 + cons_tem22

    c_prob = cp.Problem(cp.Minimize(c_obj), cons)
    c_layer = CvxpyLayer(c_prob, parameters=[c_p_A,c_p_B,c_p_F], variables=[c_v_tem,c_v_q,c_v_tem_u,c_v_tem_l])
    return c_v_tem_u,c_v_tem_l,c_v_tem,c_v_q,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2

def objective(fre_opt,SS_Y, SS_P, c_v_q, c_v_tem_u, c_v_tem_l):
    c_PI = 3.60
    c_time_p = int(96/fre_opt)
    c_time = int(96/fre_opt)+1
    c_0_price = np.array(pd.read_excel('Archive_Data/Price_signal.xlsx'),dtype=float)
    c_price = np.zeros((c_time_p,1))
    for i in range(24):
        for j in range(int(c_time_p/24)):
            c_price[i*4+j,0] = 0.001*c_0_price[i,1]
    c_Pmax,c_Pmin,c_tmax,c_tmin = SS_P.data_max_[0],SS_P.data_min_[0],SS_Y.data_max_[0],SS_Y.data_min_[0]
    c_tem_cost_u,c_tem_cost_l = 0.8*(c_tmax-c_tmin), 0.5*(c_tmax-c_tmin)   #true cost*（max-min)
    
    obj = sum(c_price[t,0]*(1/c_PI)*(0.25*fre_opt)*((c_Pmax-c_Pmin)*c_v_q[t]+c_Pmin) for t in range(c_time_p)) \
        +sum(c_tem_cost_u*c_v_tem_u[t] + c_tem_cost_l*c_v_tem_l[t] for t in range(c_time))    
    return obj
