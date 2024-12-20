import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx


def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,11))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
'''
0   FGM：投篮命中次数（Field Goals Made）
1   FG_PCT：投篮命中率（Field Goal Percentage）(丢弃)
2   FG3M：三分球命中次数（3-Point Field Goals Made）
3   FG3_PCT：三分球命中率（3-Point Field Goal Percentage）(丢弃)
4   FTM：罚球命中次数（Free Throws Made）
5   FT_PCT：罚球命中率（Free Throw Percentage）(丢弃)
6   OREB：进攻篮板球（Offensive Rebounds）
7   DREB：防守篮板球（Defensive Rebounds）
8   AST：助攻（Assists）
9   TOV：失误（Turnovers）
10  PF：犯规（Personal Fouls）
'FGM', 'FG_PCT', 'FG3M', 'FG3_PCT', 'FTM', 'FT_PCT', 'OREB''DREB', 'AST', 'TOV', 'PF'
0         1        2        3        4         5        6     7       8      9     10
'''
def bivariate_loss(V_pred, V_trgt):
    #mux, muy, sx, sy, corr
    # 确保V_pred和V_trgt形状相同
    # assert V_pred.shape == V_trgt.shape
    # 计算x和y方向的差值
    FGM = V_trgt[:,:,0]- V_pred[:,:,0]
    FG_PCT = V_trgt[:,:,1]- V_pred[:,:,1]
    FG3M = V_trgt[:, :, 2] - V_pred[:, :, 2]
    FG3_PCT = V_trgt[:, :, 3] - V_pred[:, :, 3]
    FTM = V_trgt[:, :, 4] - V_pred[:, :, 4]
    FT_PCT = V_trgt[:, :, 5] - V_pred[:, :, 5]
    OREB = V_trgt[:, :, 6] - V_pred[:, :, 6]
    DREB = V_trgt[:, :, 7] - V_pred[:, :, 7]
    AST = V_trgt[:, :, 8] - V_pred[:, :, 8]
    TOV = V_trgt[:, :, 9] - V_pred[:, :, 9]
    PF = V_trgt[:, :, 10] - V_pred[:, :, 10]
    # 计算s
    sFGM = torch.std(V_pred[:,:,0], dim=0)
    sFG_PCT = torch.std(V_pred[:,:,1], dim=0)
    sFG3M = torch.std(V_pred[:,:,2], dim=0)
    sFG3_PCT = torch.std(V_pred[:,:,3], dim=0)
    sFTM = torch.std(V_pred[:,:,4], dim=0)
    sFT_PCT = torch.std(V_pred[:,:,5], dim=0)
    sOREB = torch.std(V_pred[:,:,6], dim=0)
    sDREB = torch.std(V_pred[:,:,7], dim=0)
    sAST = torch.std(V_pred[:,:,8], dim=0)
    sTOV = torch.std(V_pred[:,:,9], dim=0)
    sPF = torch.std(V_pred[:,:,10], dim=0)
    # 计算相关系数corr
    # corr = torch.tanh(V_pred[:,:,4]) # corr
    # 计算sx和sy的乘积sxsy
    s = sFGM * sFG_PCT * sFG3M * sFG3_PCT * sFTM * sFT_PCT * sOREB * sDREB * sAST * sTOV * sPF
    # 计算z，即误差平方和
    z = (FGM/sFGM)**2 + \
        (FG_PCT/sFG_PCT)**2 + \
        (FG3M/sFG3M)**2 + \
        (FG3_PCT/sFG3_PCT)**2 + \
        (FTM/sFTM)**2 + \
        (FT_PCT/sFT_PCT)**2 + \
        (OREB/sOREB)**2 + \
        (DREB/sDREB)**2 + \
        (AST/sAST)**2 + \
        (TOV/sTOV)**2 + \
        (PF/sPF)**2
    # 计算负相关系数
    negRho = torch.tensor(1)
    # 计算分子
    result = torch.exp(-z/(1000*negRho))
    # 计算归一化因子
    denom = 2 * np.pi * (s * torch.sqrt(negRho))
    # 计算最终的概率密度函数
    result = result / denom
    # 数值稳定性处理
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    # 计算均值
    result = torch.mean(result)
    return result


def bivariate_lossOld(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # 确保V_pred和V_trgt形状相同
    # assert V_pred.shape == V_trgt.shape
    # 计算x和y方向的差值
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]
    # 计算sx和sy
    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    # 计算相关系数corr
    corr = torch.tanh(V_pred[:, :, 4])  # corr
    # 计算sx和sy的乘积sxsy
    sxsy = sx * sy
    # 计算z，即误差平方和
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    # 计算负相关系数
    negRho = 1 - corr ** 2
    # 计算分子
    result = torch.exp(-z / (2 * negRho))
    # 计算归一化因子
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    # 计算最终的概率密度函数
    result = result / denom
    # 数值稳定性处理
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    # 计算均值
    result = torch.mean(result)
    return result
