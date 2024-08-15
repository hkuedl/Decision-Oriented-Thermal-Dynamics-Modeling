#%%
import numpy as np
import math
import pandas as pd
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
import Archive_Model_Func
import matplotlib.pyplot as plt
import time

nn_zone = 6
T_Fre = 4
Train_s = T_Fre*24*(31+28+31+30+31)
Train_e = Train_s + T_Fre*24*(30+31) + 1
Train_s2 = Train_e - 1
Train_e2 = Train_s2 + T_Fre*24*31 + 1

lack_ratio = 0.0

SS_X,SS_P,SS_Y,SS_Y_ad,S_X_tr,S_P_tr,S_Y_tr,S_Y_ad_tr,S_X_te,S_P_te,S_Y_te,S_Y_ad_te,Y_tr1,Y_te1 \
    = Archive_Model_Func.data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,nn_zone,lack_ratio)

FF_tr = 1
FF_te = 1
batch_step_bei_tr = 1
batch_step_bei_te = 1
batch_time_tr = torch.arange(0., 24*T_Fre+1, FF_tr).to(device)
batch_time_te = torch.arange(0., 24*T_Fre+1, FF_te).to(device)
NN_step_tr = (len(batch_time_tr)-1)*batch_step_bei_tr + 1
NN_step_te = (len(batch_time_te)-1)*batch_step_bei_te + 1
batch_step_tr = (batch_time_tr[1] - batch_time_tr[0])/batch_step_bei_tr
batch_step_te = (batch_time_te[1] - batch_time_te[0])/batch_step_bei_te

T_Y_tr, batch_y0, batch_y11, batch_y = Archive_Model_Func.getdata(batch_step_bei_tr, T_Fre,S_Y_tr,S_X_tr,S_P_tr,S_Y_ad_tr, FF_tr, nn_zone)
T_Y_te, batch_t_y0, batch_t_y11, batch_t_y = Archive_Model_Func.getdata(batch_step_bei_te, T_Fre,S_Y_te,S_X_te,S_P_te,S_Y_ad_te, FF_te, nn_zone)


#%% first: train
y_linear = 0   #0: Neural ODE;  1: RC-based model
y_phy_ode = 1  #0: only MSE-based; 1: MSE&physics
func = Archive_Model_Func.NeuralODE_ab(nn_zone, y_linear, y_phy_ode).to(device)
optimizer = optim.Adam(func.parameters(), lr=1e-3)
epochs = 2000
epoch_freq = 100
batch_size = 16
batch_num = math.ceil(batch_y0.shape[0]/batch_size)
n_batch_list_all = []
start_time = time.time()
for epoch in range(epochs):
    batch_list = list(range(batch_y0.shape[0]))
    for num in range(batch_num):
        n_batch_list = random.sample(batch_list,min(batch_size,len(batch_list)))
        batch_list = [x for x in batch_list if x not in n_batch_list]
        n_batch_list_all.append(n_batch_list)
        n_batch_y0 = batch_y0[n_batch_list,:,:]
        n_batch_y11 = batch_y11[:,n_batch_list,:,:]
        n_batch_y = batch_y[:,n_batch_list,:,:]
        optimizer.zero_grad()
        y_input,y_phy = 1,1  #with/without inout in ODE; output physics-related terms
        n_pred_y,n_pred_comp2,n_pred_comphy,n_pred_comphy_ori = odeint(func, n_batch_y0, n_batch_y11, y_input, y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
        n_pred_y.to(device)
        n_pred_comp2.to(device)
        n_pred_comphy.to(device)
        n_pred_comphy_ori.to(device)
        loss1 = torch.mean((n_pred_y - n_batch_y)**2)
        loss2 = torch.mean(n_pred_comphy ** 2)
        if y_phy_ode == 1:
            loss = loss1 + 5*loss2
        elif y_phy_ode == 0:
            loss = loss1
        loss.backward()
        optimizer.step()
    if epoch % epoch_freq == 0:
        with torch.no_grad():
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(epoch,(num+1)*batch_size,batch_y0.shape[0],loss.item(),loss1.item(),loss2.item()))

end_time = time.time()

train_time = end_time - start_time
print(train_time)

pred_y,pred_comp2,pred_comphy,pred_comphy_ori = odeint(func, batch_y0, batch_y11, y_input, y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
pred_y.to(device)
pred_comp2.to(device)
pred_comphy.to(device)
pred_comphy_ori.to(device)
loss_tr1 = torch.mean((pred_y - batch_y)**2)
loss_tr2 = torch.mean((pred_comphy)**2)
print('Train loss1:')
print(loss_tr1)
print('Train loss phy:')
print(loss_tr2)

pred_t_y,pred_t_comp2,pred_t_comphy,pred_t_comphy_ori = odeint(func, batch_t_y0, batch_t_y11, y_input, y_phy, batch_time_te, method = 'euler', options = {'step_size': batch_step_te})
pred_t_y.to(device)
pred_t_comp2.to(device)
pred_t_comphy.to(device)
pred_t_comphy_ori.to(device)
loss_te1 = torch.mean((pred_t_y - batch_t_y)**2)
loss_te2 = torch.mean(pred_t_comphy**2)
print('Test loss1:')
print(loss_te1)
print('Test loss phy:')
print(loss_te2)
def to_np(x):
    return x.cpu().detach().numpy()

T_Y_tr_ = to_np(T_Y_tr)
pred_y_ = np.squeeze(to_np(pred_y))
batch_time_tr_ = to_np(batch_time_tr)
T_Y_te_ = to_np(T_Y_te)
pred_t_y_ = np.squeeze(to_np(pred_t_y))
batch_time_te_ = to_np(batch_time_te)

#recovery temperature
test_pred = pred_t_y_.T.flatten().reshape(-1,1)
test_pred_hy = SS_Y.inverse_transform(test_pred)
Y_te10 = np.zeros((batch_t_y0.shape[0]*len(batch_time_te),1))
for i in range(batch_t_y0.shape[0]):
   for j in range(len(batch_time_te)):
       Y_te10[i*len(batch_time_te)+j,0] = Y_te1[i*97+int(batch_time_te[j]),0]
test_Err1,test_Err2,test_Errday = Archive_Model_Func.verify_test(test_pred_hy,Y_te10,batch_t_y.shape[0])

train_pred = pred_y_.T.flatten().reshape(-1,1)
train_pred_hy = SS_Y.inverse_transform(train_pred)
Y_tr10 = np.zeros((batch_y0.shape[0]*len(batch_time_tr),1))
for i in range(batch_y0.shape[0]):
   for j in range(len(batch_time_tr)):
       Y_tr10[i*len(batch_time_tr)+j,0] = Y_tr1[i*97+int(batch_time_tr[j]),0]
train_Err1,train_Err2,train_Errday = Archive_Model_Func.verify_test(train_pred_hy,Y_tr10,batch_y.shape[0])

#torch.save(func.state_dict(), 'Archive_NNfile/pretrain_phy_'+str(nn_zone)+'_00.pt')

#%% then: optimization
y_linear = 0
y_phy_ode = 1
y_input,y_phy = 1,1
m_state_dict = torch.load('Archive_NNfile/pretrain_phy_6.pt')
func_new = Archive_Model_Func.NeuralODE_ab(nn_zone, y_linear, y_phy_ode).to(device)
func_new.load_state_dict(m_state_dict)
pred_y,pred_comp2,pred_comphy,pred_comphy_ori = odeint(func_new, batch_y0, batch_y11, y_input,y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
pred_y.to(device)
pred_comp2.to(device)
pred_comphy.to(device)
pred_comphy_ori.to(device)
pred_t_y,pred_t_comp2,pred_t_comphy,pred_t_comphy_ori = odeint(func_new, batch_t_y0, batch_t_y11, y_input,y_phy, batch_time_te, method = 'euler', options = {'step_size': batch_step_te})
pred_t_y.to(device)
pred_t_comp2.to(device)
pred_t_comphy.to(device)
pred_t_comphy_ori.to(device)

fre_opt = 1    #time interval:1:15min, 2:30min, 4:1h;
fre_cal = 1   #time interval:1:15min, 3:5min

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
c_p_A,c_p_B,c_p_F = cp.Parameter(),cp.Parameter(),cp.Parameter(c_time_dd-1)
c_v_tem,c_v_q = cp.Variable(c_time_dd),cp.Variable(c_time_p)
c_v_tem_u,c_v_tem_l = cp.Variable(c_time, pos=True),cp.Variable(c_time, pos=True)

c_upper_tem = np.zeros((c_time,1))
c_lower_tem = np.zeros((c_time,1))
c_upper_p = 60*np.ones((c_time_p,1))
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
c_PI = 3.6
c_upper_q = SS_P.transform(c_PI*c_upper_p).ravel()
c_lower_q = SS_P.transform(c_PI*c_lower_p).ravel()

c_initial_tem = c_lower_tem[0]

c_Pmax,c_Pmin,c_tmax,c_tmin = SS_P.data_max_[0],SS_P.data_min_[0],SS_Y.data_max_[0],SS_Y.data_min_[0]
c_tem_cost_u,c_tem_cost_l = 0.8*(c_tmax-c_tmin), 0.5*(c_tmax-c_tmin)   #true cost*（max-min)

c_obj1 = sum(c_price[t,0]*(1/c_PI)*(0.25*fre_opt)*((c_Pmax-c_Pmin)*c_v_q[t]+c_Pmin) for t in range(c_time_p))
c_obj2 = sum(c_tem_cost_u*c_v_tem_u[t] + c_tem_cost_l*c_v_tem_l[t] for t in range(c_time))

c_obj = sum(c_price[t,0]*(1/c_PI)*(0.25*fre_opt)*((c_Pmax-c_Pmin)*c_v_q[t]+c_Pmin) for t in range(c_time_p)) \
    +sum(c_tem_cost_u*c_v_tem_u[t] + c_tem_cost_l*c_v_tem_l[t] for t in range(c_time))

cons_tem1 = [c_v_tem[fre_cal*t] <= c_upper_tem[t]+c_v_tem_u[t] for t in range(c_time)]
cons_tem2 = [c_v_tem[fre_cal*t] >= c_lower_tem[t]-c_v_tem_l[t] for t in range(c_time)]
cons_tem11 = [c_v_tem_u[t] <= 0.5 for t in range(c_time)]
cons_tem22 = [c_v_tem_l[t] <= 0.5 for t in range(c_time)]
cons_q = [c_v_q <= c_upper_q, c_v_q >= c_lower_q]
cons_model = [c_v_tem[t+1] == c_v_tem[t] + (1/fre_cal)*(c_p_A*c_v_tem[t] + c_p_B*c_v_q[int(t//c_dT)] + c_p_F[t]) for t in range(c_time_dd-1)]
cons_ini = [c_v_tem[0] == c_lower_tem[0]]

cons = cons_tem1 + cons_tem2 + cons_q + cons_model + cons_ini + cons_tem11 + cons_tem22

c_prob = cp.Problem(cp.Minimize(c_obj), cons)
c_layer = CvxpyLayer(c_prob, parameters=[c_p_A,c_p_B,c_p_F], variables=[c_v_tem,c_v_q,c_v_tem_u,c_v_tem_l])


def to_np(x):
    return x.cpu().detach().numpy()
Cr_obj,Cr_q,Cr_tem,Cr_temu,Cr_teml = [],[],[],[],[]
Cr_obj1,Cr_obj2 = [],[]
Ce_obj1,Ce_obj2 = [],[]
Ce_obj,Ce_q,Ce_tem,Ce_temu,Ce_teml = [],[],[],[],[]
Cr_q_hy,Cr_tem_hy,Ce_q_hy,Ce_tem_hy = [],[],[],[]
train_days = batch_y0.shape[0]
test_days = batch_t_y0.shape[0]
c_p_A.value = list(func_new.parameters())[0].item()
c_p_B.value = list(func_new.parameters())[1].item()
start_time = time.time()
for d in range(train_days):
    c_p_F.value = to_np(pred_comp2[:,d,0,0])
    c_prob.solve(solver=cp.GUROBI,verbose=False)
    Cr_obj.append(c_prob.value)
    Cr_obj1.append(c_obj1.value)
    Cr_obj2.append(c_obj2.value)
    Cr_q.append(c_v_q.value)
    Cr_q_hy.append(SS_P.inverse_transform(c_v_q.value.reshape(-1,1)).ravel())
    Cr_tem.append(c_v_tem.value)
    Cr_tem_hy.append(SS_Y.inverse_transform(c_v_tem.value.reshape(-1,1)).ravel())
    Cr_temu.append(c_v_tem_u.value)
    Cr_teml.append(c_v_tem_l.value)
    print("status:", c_prob.status)
end_time = time.time()
Opt_time = end_time - start_time
print(Opt_time)
for d in range(test_days):
    c_p_F.value = to_np(pred_t_comp2[:,d,0,0])
    c_prob.solve(solver=cp.GUROBI,verbose=False)
    Ce_obj.append(c_prob.value)
    Ce_obj1.append(c_obj1.value)
    Ce_obj2.append(c_obj2.value)
    Ce_q.append(c_v_q.value)
    Ce_q_hy.append(SS_P.inverse_transform(c_v_q.value.reshape(-1,1)).ravel())
    Ce_tem.append(c_v_tem.value)
    Ce_tem_hy.append(SS_Y.inverse_transform(c_v_tem.value.reshape(-1,1)).ravel())
    Ce_temu.append(c_v_tem_u.value)
    Ce_teml.append(c_v_tem_l.value)
    print("status:", c_prob.status)

Cr_objsum = sum(Cr_obj)
Ce_objsum = sum(Ce_obj)
Cr_objsum1 = sum(Cr_obj1)
Ce_objsum1 = sum(Ce_obj1)
Cr_objsum2 = sum(Cr_obj2)
Ce_objsum2 = sum(Ce_obj2)
print('Train objective:')
print(Cr_objsum)
print('Test objective:')
print(Ce_objsum)

#%% save results
import pandas as pd
writer = pd.ExcelWriter('Archive_Results/6_tem_q_ori_new.xlsx')
to_Cr_q = pd.DataFrame(Cr_q_hy)
to_Cr_q.to_excel(writer,sheet_name='q_tr',index=False)
to_Cr_tem = pd.DataFrame(Cr_tem_hy)
to_Cr_tem.to_excel(writer,sheet_name='tem_tr',index=False)
to_Ce_q = pd.DataFrame(Ce_q_hy)
to_Ce_q.to_excel(writer,sheet_name='q_te',index=False)
to_Ce_tem = pd.DataFrame(Ce_tem_hy)
to_Ce_tem.to_excel(writer,sheet_name='tem_te',index=False)
#writer.save()
writer.close()