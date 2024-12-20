import csv
import os
import math
import sys

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, default_collate
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)


def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


'''
data是一个包含NBA球队数据的DataFrame，包括每个球队的命中率、得分、失分和胜率等信息。
函数首先从数据中提取特征向量和标签，并根据这些数据创建节点特征矩阵和邻接矩阵。
特别地，节点特征矩阵的每一行对应一个球队，每一列对应一个特征（命中率、得分、失分）。
邻接矩阵表示每两个球队之间的关系，边的权重为它们的得分之比。
'''

def match_to_graph(seq_, seq_rel,ratio_dict, norm_lap_matr=True):
    DEBUG = True
    try:
        if DEBUG:
            raise FileNotFoundError
        print("LOADING EXIST V,A TENSOR......")
        V = torch.load('Vsave.pt')
        A = torch.load('Asave.pt')
        return V,A
    except(FileNotFoundError):
        seq_ = seq_.squeeze()
        # seq_rel = seq_rel.squeeze()
        seq_len = seq_.shape[2]
        max_nodes = seq_.shape[0]

        V = np.zeros((seq_len, max_nodes, 8))
        A = np.zeros((seq_len, max_nodes, max_nodes))
        for s in range(seq_len):
            step_ = seq_[:, :, s]
            # step_rel = seq_rel[:, :, s]
            for h in range(len(step_)):
                V[s, h, :] = step_[h]
                A[s, h, h] = 1
                for k in range(h + 1, len(step_)):
                    if (A[s, h, k] != 0):  # seq_len天数范围里的第一天不要去处理
                        A[s, h, k] = A[s, h, k] * 0.5 + ratio_dict[(h, k)] * 0.5
                        A[s, k, h] = A[s, h, k] * 0.5 + ratio_dict[(h, k)] * 0.5

                    else:
                        # print("s,h,k:", s, h, k)
                        A[s, h, k] = 0 if (h, k) not in ratio_dict else ratio_dict[(h, k)]
                        A[s, k, h] = 0 if (h, k) not in ratio_dict else ratio_dict[(h, k)]
            if norm_lap_matr:
                G = nx.from_numpy_matrix(A[s, :, :])
                A[s, :, :] = np.abs(nx.normalized_laplacian_matrix(G).toarray())
        V = torch.from_numpy(V).type(torch.float)
        A = torch.from_numpy(A).type(torch.float)
        torch.save(V, 'Vsave.pt')
        torch.save(A, 'Asave.pt')
        return V,A

def match_to_graph_time(seq_, seq_rel,ratio_dict, norm_lap_matr=True):
    DEBUG = True
    initialValue = 0.5
    try:
        if DEBUG:
            raise FileNotFoundError
        print("LOADING EXIST V,A TENSOR......")
        V = torch.load('Vsave.pt')
        A = torch.load('Asave.pt')
        return V,A
    except(FileNotFoundError):
        seq_ = seq_.squeeze()
        # seq_rel = seq_rel.squeeze()
        seq_len = seq_.shape[2]
        max_nodes = seq_.shape[0]

        V = np.zeros((seq_len, max_nodes, 8))
        A = np.zeros((seq_len, max_nodes, max_nodes))
        # fill the A with 0.5
        A.fill(initialValue)
        for s in range(seq_len):
            step_ = seq_[:, :, s]
            # step_rel = seq_rel[:, :, s]
            for h in range(len(step_)):
                V[s, h, :] = step_[h]
                A[s, h, h] = 1
                for k in range(h + 1, len(step_)):
                    if (A[s, h, k] != initialValue):
                        A[s, h, k] = A[s, h, k] * initialValue + ratio_dict[(h, k)] * 0.5
                        A[s, k, h] = A[s, h, k] * initialValue + ratio_dict[(h, k)] * 0.5
                    else:
                        A[s, h, k] = initialValue if (h, k) not in ratio_dict else ratio_dict[(h, k)]
                        A[s, k, h] = initialValue if (h, k) not in ratio_dict else ratio_dict[(h, k)]
            if norm_lap_matr:
                G = nx.from_numpy_matrix(A[s, :, :])
                A[s, :, :] = np.abs(nx.normalized_laplacian_matrix(G).toarray())
            if s!=0:
                for h in range(len(step_)):
                    for k in range(h + 1, len(step_)):
                        if (A[s, h, k] != initialValue):
                            A[s, h, k] = A[s, h, k] * 0.5 + A[s - 1, h, k] * 0.5
                            A[s, k, h] = A[s, h, k] * 0.5 + A[s - 1, h, k] * 0.5
                        else:
                            A[s, h, k] = A[s - 1, h, k]
                            A[s, k, h] = A[s - 1, h, k]
        V = torch.from_numpy(V).type(torch.float)
        A = torch.from_numpy(A).type(torch.float)
        torch.save(V, 'Vsave.pt')
        torch.save(A, 'Asave.pt')
        return V,A

# 将NBA球队数据转换为图形结构
# seq_代表最近5场
# def match_to_graphOld(seq_nba,match_num,ratio_dict ,norm_lap_matr=True):
#     DEBUG = True
#     try:
#         if DEBUG:
#             raise FileNotFoundError
#         V = torch.load('Vsave.pt')
#         A = torch.load('Asave.pt')
#         return V,A
#     except(FileNotFoundError):
#         # seq_match = seq_match.squeeze()
#         seq_nba = seq_nba.squeeze()
#         seq_len = seq_nba.shape[0] #考虑的比赛场数范围（赛季长度）以及比赛结果范围(长的那个) 30
#         max_nodes = match_num #全部比赛场数 30?
#         scale = 0.5
#         # seq_nba = torch.permute(seq_nba,(1,0))
#         seq_match_len = match_num #比赛结果的个数
#         V = np.zeros((seq_len, len(seq_nba[0,:][0,:]), 11))#在各比赛的特征信息
#         A = np.zeros((seq_len, max_nodes, max_nodes))# 邻接矩阵
#         for s in range(seq_len):
#             step_ = seq_nba[s,:] #第s个赛季的全部队伍全部特征
#             for h in range(len(step_[0,:])):
#                 V[s, h, :] = step_[:,h]
#
#             for h in range(max_nodes):#h代表第s个赛季比赛队伍编号，len(step_)就是这个赛季的比赛队伍总数
#                 A[s, h, h] = 1
#                 for k in range(h + 1, max_nodes):
#                     # l2_norm = anorm(step_rel[h], step_rel[k])
#                     #第s天比赛的所有队伍在邻接矩阵中填写比分
#                     if(A[s, h, k]!=0): #seq_len天数范围里的第一天不要去处理
#                         A[s, h, k] = A[s, h, k] * 0.5 + ratio_dict[(h,k)]*0.5
#                         A[s, k, h] = A[s, h, k] * 0.5 + ratio_dict[(h,k)]*0.5
#
#                     else:
#                         # print("s,h,k:", s, h, k)
#                         A[s, h, k] = 0 if (h,k) not in ratio_dict else ratio_dict[(h,k)]
#                         A[s, k, h] = 0 if (h,k) not in ratio_dict else ratio_dict[(h,k)]
#             if norm_lap_matr:
#                 G = nx.from_numpy_matrix(A[s, :, :])
#                 A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()
#         V = torch.from_numpy(V).type(torch.float)
#         A = torch.from_numpy(A).type(torch.float)
#         torch.save(V,'Vsave.pt')
#         torch.save(A,'Asave.pt')
#         return V,A
#         # return V, \
#         #        torch.from_numpy(A).type(torch.float)




def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def my_collate(batch):
    obs_traj = [item[0] for item in batch]
    pred_traj_gt = [item[1] for item in batch]
    loss_mask = [item[2] for item in batch]
    V_obs = [item[3] for item in batch]
    A_obs = [item[4] for item in batch]
    V_tr = [item[5] for item in batch]
    A_tr = [item[6] for item in batch]
    names = [item[7] for item in batch]  # 每个batch的参数
    ids = [item[8] for item in batch]

    obs_traj = torch.stack(obs_traj, dim=0)
    pred_traj_gt = torch.stack(pred_traj_gt, dim=0)
    loss_mask = torch.stack(loss_mask, dim=0)
    V_obs = torch.stack(V_obs, dim=0)
    A_obs = torch.stack(A_obs, dim=0)
    V_tr = torch.stack(V_tr, dim=0)
    A_tr = torch.stack(A_tr, dim=0)

    return obs_traj, pred_traj_gt, loss_mask, V_obs, A_obs, V_tr, A_tr, names, ids

class NBADatasetOld(Dataset):
    def __init__(self, data_dir, skip=1, min_team=1, delim='\t',norm_lap_matr = True):
        # self.data = pd.read_csv(data_dir)
        self.max_peds_in_frame = 0  # 每帧中的最大行人数，初始化为0
        self.data_dir = data_dir  # 数据文件夹路径

        self.skip = skip  # 序列采样步长
        self.delim = delim  # 数据文件的分隔符
        self.norm_lap_matr = norm_lap_matr  # 归一化拉普拉斯矩阵
        self.seq_len = []
        team_dict = {}
        self.adj_mx_list = []
        allTeams=[]
        self.home_features = []
        self.away_features = []
        self.seq_list=[]
        self.seq_match = []
        self.ratios =[]
        self.ratio_dict={}
        all_files = os.listdir(self.data_dir)  # 列出数据文件夹中的所有文件
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # 获得所有文件的完整路径

        for (index,path) in enumerate(all_files):  # 遍历所有文件
            data = pd.read_csv(path)
            self.seq_len.append(len(data) * 2)  # 序列长度

            data.sort_values(by=['GAME_DATE'], inplace=True)
            team_home = data['TEAM_ABBREVIATION_HOME'].unique()
            team_away = data['TEAM_ABBREVIATION_AWAY'].unique()
            for i, team in enumerate(team_home):
                team_dict[str(team)] = i
            home_team = data['TEAM_ABBREVIATION_HOME']
            away_team = data['TEAM_ABBREVIATION_AWAY']

            score_ratio = data['ratio']
            for i, team in home_team.iteritems():
                if (team_dict[home_team[i]],team_dict[away_team[i]]) not in self.ratio_dict:
                    self.ratio_dict[(team_dict[home_team[i]],team_dict[away_team[i]])] = []
                    self.ratio_dict[(team_dict[home_team[i]], team_dict[away_team[i]])] = score_ratio[i]
                else:
                    self.ratio_dict[(team_dict[home_team[i]], team_dict[away_team[i]])] = score_ratio[i] * 0.5 + 0.5 * self.ratio_dict[(team_dict[home_team[i]], team_dict[away_team[i]])]

            home_feature = data[
                ['FGM_HOME', 'FG3M_HOME', 'FTM_HOME', 'OREB_HOME',
                 'DREB_HOME', 'AST_HOME', 'TOV_HOME', 'PF_HOME']].values
            away_feature = data[
                ['FGM_AWAY', 'FG3M_AWAY', 'FTM_AWAY', 'OREB_AWAY',
                 'DREB_AWAY', 'AST_AWAY', 'TOV_AWAY', 'PF_AWAY']].values

            self.ratios.append(score_ratio)
            home_feature = home_feature.astype('float32')
            home_feature = torch.tensor(home_feature)
            away_feature = away_feature.astype('float32')
            away_feature = torch.tensor(away_feature)
            # print(home_feature)
            # print(self.home_features)
            self.home_features.append(torch.transpose(home_feature,0,1))
            self.away_features.append(torch.transpose(away_feature,0,1))

            # self.home_features.append(torch.cat((torch.transpose(home_feature,0,1), torch.tensor([score_ratio]).float()), dim=0))
            # self.away_features.append(torch.cat((torch.transpose(away_feature,0,1), torch.tensor([1 / score_ratio]).float()), dim=0))


            # 合并主场和客场队伍名称
            allTeams.append(pd.concat([pd.Series(team_home), pd.Series(team_away)], ignore_index=True))

            # 获取不同的队伍数量
            # num_teams = len(allTeams[index].unique())

            # curr_seq = np.zeros(num_teams, 2, self.seq_len)  # 初始化轨迹
            temp1=[]
            for match in home_feature:
               temp1.append(match)
            self.seq_list.append(temp1)
            # self.seq_list=torch.stack(self.seq_list[index])
            # for _, ped_id in enumerate(allTeams):
            #     seq_list.append(home_feature[index].unsqueeze(0))
            # print()
            # 开始制作seq_match
            # home_team是1189个有重复的主场列表
            # matches='球队1':[
            #     比赛1结果，比赛2结果，比赛3结果，比赛4结果，比赛5结果
            #   ],结果是ratio值
            matches={}
            for i, team in home_team.iteritems():
                if team not in matches:
                    matches[team] = []
                matches[team].append(self.ratios[index][i])
            # 现在matches是个遍历了所有比赛的字典，键为队名，值为比赛结果
            # print(len(self.seq_match),index)
            temp=[]
            for i in matches:
                temp.append(matches[i])
            max_length = max(len(elem) for elem in temp)  # 找到最长元素的长度
            # 将所有元素填充到相同的长度
            for i in range(len(temp)):
                temp[i] = temp[i] + [0] * (max_length - len(temp[i]))
            temp = torch.tensor(temp)
            self.seq_match.append(temp)


        max_length = max(len(sublist) for sublist in self.seq_list)
        # 遍历所有子列表，填充0
        for sublist in self.seq_list:
            while len(sublist) < max_length:
                sublist.append(torch.zeros_like(sublist[0]))

        tensor_list = [torch.stack(sublist) for sublist in self.seq_list]
        final_tensor = torch.stack(tensor_list, dim=0)

        self.seq_list = final_tensor

        max_len = max([tensor.size(1) for tensor in self.seq_match])
        # 首先将每个Tensor在两个维度上进行补齐
        padded_data = [torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), 'constant', 0) for tensor in self.seq_match]
        padded_data = torch.nn.utils.rnn.pad_sequence(padded_data, batch_first=True)
        # 然后将补齐后的Tensor进行拼接
        new_tensor = torch.stack([padded_data[i] for i in range(padded_data.shape[0])])

        self.seq_match = new_tensor

        max_len = max([len(s.values) for s in self.ratios])
        #填充ratio
        padded_series_list = []
        for series in self.ratios:
            padded_series = np.pad(series, (0, max_len - len(series)), 'constant', constant_values=0)
            padded_series_list.append(padded_series)
        tensor_data = torch.stack([torch.Tensor(arr) for arr in padded_series_list])
        self.ratios = tensor_data

        # 填充 home_features
        padded_data = [torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), 'constant', 0) for tensor in self.home_features]
        padded_data = torch.nn.utils.rnn.pad_sequence(padded_data, batch_first=True)
        concat_tensor = torch.cat((padded_data,), dim=0)
        self.home_features = concat_tensor

        # 填充 away_features
        padded_data = [torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), 'constant', 0) for tensor in self.away_features]
        padded_data = torch.nn.utils.rnn.pad_sequence(padded_data, batch_first=True)
        concat_tensor = torch.cat((padded_data,), dim=0)
        self.away_features = concat_tensor

        #计算每个赛季的起始和结束球队编号，存储到 seq_start_end 列表中
        self.seq_start_end = [
            (0, end-1)
            for end in self.seq_len
        ]

        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]
            # print("[DEBUG]seq_", self.seq_list[start:end, :])
            # print("[DEBUG]seq_rel", self.obs_traj_rel[start:end, :])
            #传入1189场比赛以及他们编号对应的主场球队
            v_, a_ = match_to_graph(self.seq_list[ss,:,:], self.seq_match[ss,:,:],self.ratio_dict, norm_lap_matr=self.norm_lap_matr)

            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = match_to_graph(self.seq_list[ss, :,:], self.seq_match[ss,:,:],self.ratio_dict, norm_lap_matr=self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()


    def __len__(self):
        sum = 0
        for length in self.seq_len:
            sum+=length
        return sum

    def __getitem__(self, index):
        # game = self.data.iloc[index]
        # sample = {'input': (home_features, away_features), 'adj_mx': self.adj_mx_list[game[0]]}
        # target = (self.team_dict[str(home_team)], self.team_dict[str(away_team)])
        # return sample, target
        # print("now get index:",index)
        index = index % 21
        return self.ratios[index],self.home_features[index], self.away_features[index],self.v_obs[index], self.A_obs[index],self.v_pred[index], self.A_pred[index]

'''
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里
*   这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里这里

'''


class NBADataset(Dataset):
    def __init__(
        self, data_dir, obs_len=5, pred_len=5, skip=1, threshold=0.002,
        norm_lap_matr = False):
        """
        Args:
        - data_dir: 数据集文件所在的目录，文件格式为 <frame_id> <ped_id> <x> <y>
        - obs_len: 输入轨迹的时间步数
        - pred_len: 输出轨迹的时间步数
        - skip: 创建数据集时跳过的帧数
        - threshold: 当使用线性预测器时，被视为非线性轨迹的最小误差
        - min_ped: 序列中应该存在的最小行人数量
        - delim: 数据集文件中的分隔符
        """
        super(NBADataset, self).__init__()

        self.max_teams_in_frame = 0  # 每帧中的最大行人数，初始化为0
        self.data_dir = data_dir  # 数据文件夹路径
        self.features_dir = data_dir + "../ba"
        self.obs_len = obs_len  # 观测序列长度
        self.pred_len = pred_len  # 预测序列长度
        self.skip = skip  # 序列采样步长
        self.seq_len = self.obs_len + self.pred_len  # 序列长度
        self.norm_lap_matr = norm_lap_matr  # 归一化拉普拉斯矩阵
        self.ratio_dict={}
        team_dict = {}
        all_files = os.listdir(self.data_dir)  # 列出数据文件夹中的所有文件
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # 获得所有文件的完整路径
        all_feature_files = os.listdir(self.features_dir)  # 列出数据文件夹中的所有文件
        all_feature_files = [os.path.join(self.features_dir, _path) for _path in all_feature_files]  # 获得所有文件的完整路径

        num_peds_in_seq = []  # 用于记录每个序列中的行人数目
        seq_list = []  # 用于存储轨迹数据
        seq_name_list = []  # 用于存储相对坐标轨迹数据
        seq_id_list = []
        # self.seq_match = []
        ratios = []
        loss_mask_list = []  # 用于存储损失掩码数据
        # non_linear_ped = []  # 用于记录轨迹是否是非线性的
        for index,path in enumerate(all_files):  # 遍历所有文件
            rawData = pd.read_csv(path)
            home_feature = rawData[
                ['TEAM_ABBREVIATION_HOME',"id",'FGM_HOME', 'FG3M_HOME', 'FTM_HOME', 'OREB_HOME',
                 'DREB_HOME', 'AST_HOME', 'TOV_HOME', 'PF_HOME']].values
            away_feature = rawData[
                ['TEAM_ABBREVIATION_AWAY',"id",'FGM_AWAY', 'FG3M_AWAY', 'FTM_AWAY', 'OREB_AWAY',
                 'DREB_AWAY', 'AST_AWAY', 'TOV_AWAY', 'PF_AWAY']].values
            team_feature={}
            for feature in (home_feature, away_feature):
                for row in feature:
                    team = row[0]
                    id = row[1]
                    values = row[2:]
                    if team not in team_feature:
                        team_feature[team] = [id] + values
                    else:
                        team_feature[team] = [id] + [(a + b) / 2 for a, b in zip(team_feature[team], values)]

            team_home = rawData['TEAM_ABBREVIATION_HOME']
            team_away = rawData['TEAM_ABBREVIATION_AWAY']
            score_ratio = rawData['ratio']
            for i, team in enumerate(team_home):
                if str(team) not in team_dict:
                    team_dict[str(team)] = i
            for i, team in team_home.iteritems():
                if (team_dict[team_home[i]],team_dict[team_away[i]]) in self.ratio_dict:
                    self.ratio_dict[(team_dict[team_home[i]], team_dict[team_away[i]])] = \
                        score_ratio[i] * 0.5 + 0.5 * self.ratio_dict[(team_dict[team_home[i]], team_dict[team_away[i]])]
                else:
                    self.ratio_dict[(team_dict[team_home[i]],team_dict[team_away[i]])] = []
                    self.ratio_dict[(team_dict[team_home[i]], team_dict[team_away[i]])] = score_ratio[i]


            # TODO:ratio平均了可能会导致比赛胜负预测有点问题
            matches = rawData.groupby('TEAM_ABBREVIATION_HOME').mean()['ratio']
            ratios.append(score_ratio)

            # temp = []
            # for i in matches:
            #     temp.append(i)
            # temp = torch.tensor(temp)
            # self.seq_match.append(temp)
            offset = 0
            if self.data_dir == './datasets/match/train/':
                offset = 0
            elif self.data_dir == './datasets/match/val/':
                # 跳过 06 -16
                offset = 11
            elif self.data_dir == './datasets/match/test/':
                offset = 14
            data = pd.read_csv(all_feature_files[index+offset])
            data = data.drop(['FG_PCT','FG3_PCT','FT_PCT','GAME_DATE', 'SEASON_ID', 'HOA', 'RATIO'], axis=1)

            data = data.values.tolist()

            # frames = np.unique(data[data[0,:]]).tolist()  # 获取所有帧编号，并转换为列表形式
            maxlen=data[-1][1]
            minlen=data[0][1]
            frames = [i for i in range(minlen,maxlen)]
            # 得到一个data列表
            frame_data = []
            # 得到所有帧的所有特征frame_data
            for frame in frames:
                cur_frame=[]
                for row in data:
                    if row[1] == frame:
                        cur_frame.append(row)
                frame_data.append(cur_frame)  # 将当前帧的数据存储到列表中     #2000有118个[0-117]
            # 值为103
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))  # 计算当前文件中的序列数目
            # 范围0-104


            for idx in range(0, num_sequences * self.skip, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)  # 将当前序列的所有帧的数据合并到一起
                teams_in_curr_seq = np.unique(curr_seq_data[:, 2])  # 获取当前序列中的所有行人编号
                self.max_teams_in_frame = max(self.max_teams_in_frame, len(teams_in_curr_seq))  # 更新最大行人数目

                curr_seq = np.zeros((len(teams_in_curr_seq), 8, self.seq_len))  # 初始化轨迹
                curr_seq_names = np.empty((len(teams_in_curr_seq), 2, self.seq_len), dtype=object)
                curr_seq_ids = np.empty((len(teams_in_curr_seq), self.seq_len), dtype=object)
                curr_loss_mask = np.zeros((len(teams_in_curr_seq), self.seq_len))  # 初始化损失掩码
                num_peds_considered = 0
                # _non_linear_ped = []
                for _, ped_id in enumerate(teams_in_curr_seq):
                    # 对于当前帧中的每个行人，获取其轨迹序列数据
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 2] == ped_id, :]
                    pad_front = frames.index(int(curr_ped_seq[0, 1])) - idx
                    pad_end = frames.index(int(curr_ped_seq[-1, 1])) - idx + 1

                    if pad_end - pad_front != self.seq_len :
                        continue

                    # 取出当前球队序列的特征数据，并转置为列向量
                    # 把特征顺序换一下，把自己队名和敌人队名放到前面去
                    new_order = [0, 1, 2, 11, 3, 4, 5, 6, 7, 8, 9, 10]

                    curr_ped_seq = curr_ped_seq[:,new_order]
                    curr_ped_names = np.transpose(curr_ped_seq[:, 2:4])
                    curr_ped_ids = np.transpose(curr_ped_seq[:, 0])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 4:])
                    # 将该队伍的特征值存入 curr_seq
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_names[_idx, :, pad_front:pad_end] = curr_ped_names
                    curr_seq_ids[_idx, pad_front:pad_end] = curr_ped_ids
                    # 标记该行人在当前时间序列中存在

                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                #判断当前时间序列中是否存在足够数量的行人
                # if num_peds_considered > min_ped:
                # non_linear_ped += _non_linear_ped
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_list.append(curr_seq[:num_peds_considered])
                seq_name_list.append(curr_seq_names[:num_peds_considered])
                seq_id_list.append(curr_seq_ids[:num_peds_considered])

                # seq_list_rel.append(curr_seq_rel[:num_peds_considered])
        #合并所有序列的相关信息
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_name_list = np.concatenate(seq_name_list, axis=0)
        seq_id_list = np.concatenate(seq_id_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        # seq_match = np.concatenate(seq_match,axis=0)
        # non_linear_ped = np.asarray(non_linear_ped)


        #将 numpy 数组转换为 PyTorch Tensor，并存储到类的属性中
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_traj_name = seq_name_list[:,:,:self.obs_len]
        self.obs_traj_id = seq_id_list[:,:self.obs_len]

        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        # self.seq_match = torch.from_numpy(seq_match).type(torch.float)
        # self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        #计算每个时间序列的起始和结束行人编号，存储到 seq_start_end 列表中
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        self.names = []
        self.ids = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            # 对当前序列观测轨迹和相对位置信息进行图像化处理
            v_,a_ = match_to_graph(self.obs_traj[start:end,:],self.obs_len,self.ratio_dict,self.norm_lap_matr)
            self.names.append(self.obs_traj_name[start:end,:])
            self.ids.append(self.obs_traj_id[start:end,:])
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())

            # 对当前序列预测轨迹和相对位置信息进行图像化处理
            v_,a_=match_to_graph(self.pred_traj[start:end,:],self.obs_len,self.ratio_dict,self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        #
        out = [

            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            # self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            # self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            #我们算出来的四个矩阵
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.names[index],
            self.ids[index]
        ]
        return out



'''
将数据集中的每个人的运动序列转换成相对坐标、速度和方向，并将它们添加到数据集中。
具体来说，代码先遍历数据集中所有人的ID，然后找到该人在当前序列中的数据。
接下来，代码计算该人的相对坐标、速度和方向，并将它们与相对坐标连接起来，形成一个序列。
这个序列被添加到数据集中，如果数据集大小达到了预定大小，则返回数据集。
如果没有达到预定大小，则继续遍历数据集中的下一个人的ID。
'''
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: 数据集文件所在的目录，文件格式为 <frame_id> <ped_id> <x> <y>
        - obs_len: 输入轨迹的时间步数
        - pred_len: 输出轨迹的时间步数
        - skip: 创建数据集时跳过的帧数
        - threshold: 当使用线性预测器时，被视为非线性轨迹的最小误差
        - min_ped: 序列中应该存在的最小行人数量
        - delim: 数据集文件中的分隔符
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0  # 每帧中的最大行人数，初始化为0
        self.data_dir = data_dir  # 数据文件夹路径
        self.obs_len = obs_len  # 观测序列长度
        self.pred_len = pred_len  # 预测序列长度
        self.skip = skip  # 序列采样步长
        self.seq_len = self.obs_len + self.pred_len  # 序列长度
        self.delim = delim  # 数据文件的分隔符
        self.norm_lap_matr = norm_lap_matr  # 归一化拉普拉斯矩阵

        all_files = os.listdir(self.data_dir)  # 列出数据文件夹中的所有文件
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # 获得所有文件的完整路径
        num_peds_in_seq = []  # 用于记录每个序列中的行人数目
        seq_list = []  # 用于存储轨迹数据
        seq_list_rel = []  # 用于存储相对坐标轨迹数据
        loss_mask_list = []  # 用于存储损失掩码数据
        # non_linear_ped = []  # 用于记录轨迹是否是非线性的
        for path in all_files:  # 遍历所有文件
            data = read_file(path, delim)  # 读取文件中的数据
            frames = np.unique(data[:, 0]).tolist()  # 获取所有帧编号，并转换为列表形式
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])  # 将当前帧的数据存储到列表中
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))  # 计算当前文件中的序列数目

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)  # 将当前序列的所有帧的数据合并到一起
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 获取当前序列中的所有行人编号
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))  # 更新最大行人数目
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))  # 初始化相对坐标轨迹
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))  # 初始化轨迹
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))  # 初始化损失掩码
                num_peds_considered = 0
                # _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # 对于当前帧中的每个行人，获取其轨迹序列数据
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    # 对轨迹序列数据进行四舍五入，保留小数点后四位
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # 计算当前行人序列与当前帧的时间偏移量
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    # 如果时间偏移量不等于指定的序列长度，则跳过该行人序列
                    if pad_end - pad_front != self.seq_len:
                        continue
                    # 取出当前行人序列的坐标数据，并转置为列向量
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # 将坐标值转化为相对坐标
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    # 将该行人的相对坐标和原始坐标存入 curr_seq 和 curr_seq_rel
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # 判断该行人的轨迹是否是线性的
                    # _non_linear_ped.append(
                    #     poly_fit(curr_ped_seq, pred_len, threshold))
                    # 标记该行人在当前时间序列中存在

                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                #判断当前时间序列中是否存在足够数量的行人
                if num_peds_considered > min_ped:
                    # non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
        #合并所有序列的相关信息
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        # non_linear_ped = np.asarray(non_linear_ped)

        #将 numpy 数组转换为 PyTorch Tensor，并存储到类的属性中
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        # self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        #计算每个时间序列的起始和结束行人编号，存储到 seq_start_end 列表中
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            # 对当前序列观测轨迹和相对位置信息进行图像化处理
            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)

            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())

            # 对当前序列预测轨迹和相对位置信息进行图像化处理
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        #
        out = [

            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            # self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            #我们算出来的四个矩阵
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out
