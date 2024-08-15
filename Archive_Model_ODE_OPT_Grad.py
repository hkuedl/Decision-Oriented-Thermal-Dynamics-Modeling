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

seed_value = 20
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

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

#%
from scipy.optimize import minimize, Bounds, minimize_scalar
import copy
y_linear = 0
y_phy_ode = 1  #0: only MSE-based; 1: MSE&physics
m_state_dict = torch.load('Archive_NNfile/pretrain_phy_6.pt')
func_new = Archive_Model_Func.NeuralODE_ab(nn_zone,y_linear, y_phy_ode).to(device)
func_new.load_state_dict(m_state_dict)

def to_np(x):
    return x.cpu().detach().numpy()
def GRAD_sing(func):
    grads = {}
    for name, params in func.named_parameters():
        grad = params.grad
        if grad is not None:
            grads[name] = grad
    return grads

def GRAD_readin(grads_accu):
    grad_index = []
    grads_accu1 = torch.Tensor([[0]]).to(device)
    for name in grads_accu.keys():
        start = grads_accu1.shape[0]
        grads_accu1 = torch.cat((grads_accu1,grads_accu[name].reshape(-1,1)), dim=0)
        end = grads_accu1.shape[0]
        grad_index.append([name,start-1,end-1])
    grads_accu1 = grads_accu1[1:,0]
    return grad_index, grads_accu1

def GRAD_readout(func,grad_index,grads_com):
    for name, params in func.named_parameters():
        for gg in grad_index:
            if name == gg[0]:
                if name == 'a' or name == 'b':
                    params.grad = grads_com[gg[1]]
                elif len(params.grad.shape) == 1:
                    row = params.grad.shape[0]
                    params.grad = grads_com[gg[1]:gg[2]]
                else:
                    row,col = params.grad.shape[0],params.grad.shape[1]
                    params.grad = grads_com[gg[1]:gg[2]].reshape(row,col)

def GRAD_comb(g1,g2, c):
    i_value = 1e-6
    g1 = g1/np.sqrt(g1.dot(g1).item())
    g2 = g2/np.sqrt(g2.dot(g2).item())
    
    g11 = g1.dot(g1).item()   #a.b=|a||b|cos<>
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0 = g1.clone()
    g_ang = g12/(np.sqrt(g11*g22))
    g0_norm = np.sqrt(g0.dot(g0).item() + i_value)   #||g_0||
    coef = c * g0_norm
    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22  #gw = x * g1 + (1-x) * g2
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        g_w_0 = (x*g1+(1-x)*g2).dot(g0).item()
        g_w_w = x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(g_w_w + i_value) + g_w_0
    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x
    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11 + (1-x)**2*g22 + 2*x*(1-x)*g12 + i_value)
    lmbda = coef/(gw_norm + i_value)
    g = g0 + lmbda * gw
    #g = 1.5*g1 + g2
    return g_ang, x, g

def to_np(x):
    return x.cpu().detach().numpy()

LR = 1e-3
lr_list = [LR]
optimizer = optim.Adam(func_new.parameters(),lr = LR)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)  #0.90+0.05: for 6-zone; 0.90+0.05: for 10-zone; 0.90+0.05: for 18-zones
###67_00:0.95+0.05; 67_01:0.9+0.1; 67_02:0.9+0.05; 67_03:0.9+0.05; 67_04:0.9+0.05; 67_05:0.95+0.05;
###90_00:0.9+0.1; 90_01:0.9+0.05; 90_02:0.9+0.05; 90_03:0.9+0.1; 90_04:0.9+0.05; 90_05:0.9+0.05;

epoch_freq = 1
batch_size = batch_y0.shape[0]
batch_num = math.ceil(batch_y0.shape[0]/batch_size)   #batch
epochs_new = 30
R_opt,R_acc = [],[]
R_wei,R_ang = [],[]
Len_acc,Len_opt,Len_c = [],[],[]
ALL_models = [copy.deepcopy(func_new.state_dict())]
start_time = time.time()
for epoch in range(epochs_new):
    batch_list = list(range(batch_y0.shape[0]))
    for num in range(batch_num):
        n_batch_list = random.sample(batch_list,min(batch_size,len(batch_list)))
        batch_list = [x for x in batch_list if x not in n_batch_list]
        n_batch_y0 = batch_y0[n_batch_list,:,:]
        n_batch_y11 = batch_y11[:,n_batch_list,:,:]
        n_batch_y = batch_y[:,n_batch_list,:,:]
        #(96,21,1,1), (190,21,1,1), (190,21,1,1)
        y_input,y_phy = 1,1
        n_pred_y,n_pred_comp2,n_pred_comphy,n_pred_comphy_ori = odeint(func_new, n_batch_y0, n_batch_y11, y_input, y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
        n_pred_y.to(device)
        n_pred_comp2.to(device)
        n_pred_comphy.to(device)
        n_pred_comphy_ori.to(device)
        #grad calculation
        c_p_A_value = torch.tensor(list(func_new.parameters())[0].item())
        c_p_B_value = torch.tensor(list(func_new.parameters())[1].item())
        c_obj_list = []
        for d in range(len(n_batch_list)):
            c_v_tem_u,c_v_tem_l,c_v_tem,c_v_q,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2 = Archive_Model_Func.Optimization(1,1, SS_Y, SS_P, nn_zone)
            c_solution = c_layer(c_p_A_value,c_p_B_value,n_pred_comp2[:,d,:,:].reshape(-1))
            c_obj_i = Archive_Model_Func.objective(1,SS_Y, SS_P, c_solution[1], c_solution[2], c_solution[3])
            c_obj_list.append(c_obj_i)
        bei_loss = 1
        c_obj_sum = sum(c_obj_list)
        loss1 = torch.mean((n_pred_y - n_batch_y)**2)
        loss2 = torch.mean(n_pred_comphy**2)
        loss = loss1 + 5*loss2
        optimizer.zero_grad()
        c_obj_sum.backward(retain_graph=True)
        grads_opt = GRAD_sing(func_new)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grads_accu = GRAD_sing(func_new)
        
        grad_index, grads_accu1 = GRAD_readin(grads_accu)
        grads_acc_len = np.sqrt(grads_accu1.dot(grads_accu1).item())
        Len_acc.append(grads_acc_len)
        grad_index, grads_opt1 = GRAD_readin(grads_opt)
        grads_opt_len = np.sqrt(grads_opt1.dot(grads_opt1).item())
        Len_opt.append(grads_opt_len)
        LL = epoch*batch_num + num
        if epoch == 0 and num == 0:
            c = 2/(1+np.exp(-1)) - 1
        else:
            c = 2/(1+np.exp(-1*Len_acc[0]/grads_acc_len)) - 1
        Len_c.append(c)
        com_ang, com_weight, com_grad = GRAD_comb(grads_accu1,grads_opt1, c)
        
        GRAD_readout(func_new,grad_index,com_grad)
        optimizer.step()

    ALL_models.append(copy.deepcopy(func_new.state_dict()))
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    R_opt.append(c_obj_sum.item())
    R_acc.append(loss.item())
    R_wei.append(com_weight)
    R_ang.append(com_ang)
    if epoch % epoch_freq == 0:
        with torch.no_grad():
            print('Train Epoch: {} [{}/{}]\tObj: {:.6f}\tLoss: {:.6f}'.format(epoch, (num+1) * batch_size, batch_y0.shape[0], c_obj_sum.item(), loss.item()/bei_loss))

end_time = time.time()

train_time = end_time - start_time

print(train_time)

Loss_opt = torch.tensor(R_opt)
Loss_phy = torch.tensor(R_acc)
# torch.save(Loss_opt,'Archive_NNfile/6zone_loss_opt')
# torch.save(Loss_phy,'Archive_NNfile/6zone_loss_phy')

#%%
s_label,s_font,s_legend = 14,16,14
s_line = 1.5

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)
range11 = list(range(epochs_new))
ax1.plot(range11, R_opt, color='r', marker='.', label='Obj. value')
ax1.set_xlabel('Epochs', fontsize = s_font)
ax1.legend(loc = 'upper left', prop = {'size':s_legend})
#ax1.set_ylim(2100,3000)
for tl in ax1.get_yticklabels():
    tl.set_color('r')
ax2 = ax1.twinx()
ax2.plot(range11, R_acc, color='purple',marker='.', label='Accuracy loss')
ax2.legend(loc = 'upper right', prop = {'size':s_legend})
#ax2.set_ylim(0.006,0.015)
for tl in ax2.get_yticklabels():
    tl.set_color('purple')
ax1.tick_params(labelsize=s_label)
ax2.tick_params(labelsize=s_label)
plt.show()
#fig.savefig('Archive_Results/6_convergence_new.pdf',format='pdf',dpi=600)

#%%select the final point
Se_poi = []
set_thres = 0.05
for i in range(len(R_acc)):
    if abs(R_acc[i] - R_acc[0])/R_acc[0] <= set_thres:
        Se_poi.append(i)
Se_opt = [R_opt[i] for i in Se_poi]
Se_final = R_opt.index(min(Se_opt))

s_label,s_font,s_legend = 14,16,14
s_line = 1.5
fig = plt.figure(figsize=(8,5))
plt.fill_between([(1-set_thres)*R_acc[0],(1+set_thres)*R_acc[0]],min(R_opt),R_opt[0], facecolor='pink', alpha = 0.3)
plt.scatter(R_acc[0],R_opt[0],s=80,c='b',label='Warm start')
plt.scatter(R_acc[1:],R_opt[1:],s=30,c='k')
plt.scatter(R_acc[Se_final],R_opt[Se_final],s=80,c='r',label='Final model')
plt.vlines((1-set_thres)*R_acc[0], min(R_opt), max(R_opt)+3, linestyle='dashed', color='k', linewidth = 0.7)
plt.vlines((1+set_thres)*R_acc[0], min(R_opt), max(R_opt)+3, linestyle='dashed', color='k', linewidth = 0.7)
plt.hlines(R_opt[0], 0, max(R_acc), linestyle='dashed', color='k', linewidth = 0.7)
plt.xlabel('Accuracy loss',fontsize = s_font)
plt.ylabel('Obj. value',fontsize = s_font)
plt.xlim(0.0006,0.0012)
plt.ylim(310,390)
# plt.xlim(0.0045,0.008)
# plt.ylim(800,1100)
# plt.xlim(0.006,0.009)
# plt.ylim(2540,2640)
plt.legend(loc='upper right',prop = {'size':s_legend})
plt.tick_params(labelsize=s_label)
plt.show()
#fig.savefig('Archive_Results/6_convergence_par_new.pdf',format='pdf',dpi=600)


#%  check accuracy results
func_new1 = Archive_Model_Func.NeuralODE_ab(nn_zone,y_linear, y_phy_ode).to(device)
func_new1.load_state_dict(ALL_models[Se_final])

def to_np(x):
    return x.cpu().detach().numpy()
pred_y,pred_comp2,pred_comphy,pred_comphy_ori = odeint(func_new1, batch_y0, batch_y11, y_input, y_phy, batch_time_tr, method = 'euler', options = {'step_size': batch_step_tr})
pred_y.to(device)
pred_comp2.to(device)
pred_comphy.to(device)
pred_comphy_ori.to(device)
loss_tr1 = torch.mean((pred_y - batch_y)**2)
loss_tr2 = torch.mean(pred_comphy**2)
print('Train loss1:')
print(loss_tr1)
print('Train loss phy:')
print(loss_tr2)
pred_t_y,pred_t_comp2,pred_t_comphy,pred_t_comphy_ori = odeint(func_new1, batch_t_y0, batch_t_y11, y_input, y_phy, batch_time_te, method = 'euler', options = {'step_size': batch_step_te})
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

pred_y_ = np.squeeze(to_np(pred_y))
pred_t_y_ = np.squeeze(to_np(pred_t_y))
train_pred = pred_y_.T.flatten().reshape(-1,1)
train_pred_hy = SS_Y.inverse_transform(train_pred)
train_Err1,train_Err2,train_Errday = Archive_Model_Func.verify_test(train_pred_hy,Y_tr1,pred_y.shape[0])
test_pred = pred_t_y_.T.flatten().reshape(-1,1)
test_pred_hy = SS_Y.inverse_transform(test_pred)
test_Err1,test_Err2,test_Errday = Archive_Model_Func.verify_test(test_pred_hy,Y_te1,pred_t_y.shape[0])

# print('Update key parameters:')
# print('a:')
# print(list(func_new1.parameters())[0].item())
# print('b:')
# print(list(func_new1.parameters())[1].item())

print(Se_final)
print(train_Err2)
print(test_Err2)

#%% solve optimal decisions
Cr_obj,Cr_q,Cr_tem,Cr_temu,Cr_teml = [],[],[],[],[]
Cr_obj1,Cr_obj2 = [],[]
Ce_obj1,Ce_obj2 = [],[]
Ce_obj,Ce_q,Ce_tem,Ce_temu,Ce_teml = [],[],[],[],[]
Cr_q_hy,Cr_tem_hy,Ce_q_hy,Ce_tem_hy = [],[],[],[]

train_days = batch_y0.shape[0]
test_days = batch_t_y0.shape[0]
c_v_tem_u,c_v_tem_l,c_v_tem,c_v_q,c_p_A,c_p_B,c_p_F,c_prob,c_layer,c_obj1,c_obj2 = Archive_Model_Func.Optimization(1,1, SS_Y, SS_P, nn_zone)
c_p_A.value = list(func_new1.parameters())[0].item()
c_p_B.value = list(func_new1.parameters())[1].item()
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
torch.save(func_new1.state_dict(), 'Archive_NNfile/pretrain_opt_6_new.pt')

#%% save results
import pandas as pd
writer = pd.ExcelWriter('Archive_Results/6_tem_q_new.xlsx')
to_Cr_q = pd.DataFrame(Cr_q_hy)
to_Cr_q.to_excel(writer,sheet_name='q_tr',index=False)
to_Cr_tem = pd.DataFrame(Cr_tem_hy)
to_Cr_tem.to_excel(writer,sheet_name='tem_tr',index=False)
to_Ce_q = pd.DataFrame(Ce_q_hy)
to_Ce_q.to_excel(writer,sheet_name='q_te',index=False)
to_Ce_tem = pd.DataFrame(Ce_tem_hy)
to_Ce_tem.to_excel(writer,sheet_name='tem_te',index=False)
writer.close()
