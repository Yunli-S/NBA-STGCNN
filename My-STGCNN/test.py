import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import social_stgcnn
import copy

def test():
    global loader_test,model
    model.eval()
    loss_bigls = []
    result = []
    result_A = []
    names = []
    ids = []
    step =0
    loss = nn.MSELoss()
    for batchIndex,batch in enumerate(loader_test):
        step+=1
        #Get data
        idList = batch [-1]
        nameList = batch [-2]
        nameList = nameList[0]
        nameList = nameList.transpose((2, 0, 1))
        idList = idList[0]
        batch = batch[:-2]
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch
        # loader_test.dataset.getName(batchIndex)
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,A_pred = model(V_obs_tmp,A_obs.squeeze())
        V_pred = V_pred.permute(0,2,3,1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        loss_bigls.append((loss(V_pred,V_tr)+loss(A_pred, A_tr))/2)
        result.append(V_pred)
        result_A.append(A_pred)
        names.append(nameList)
        ids.append(idList)

    loss_ = float(sum(loss_bigls)/len(loss_bigls))
    return loss_,result,result_A,names,ids


paths = ['./checkpoint/*my-stgcnn*']
KSTEPS=20

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)




for feta in range(len(paths)):
    loss_ls = []
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f: 
            cm = pickle.load(f)
        print("Stats:",cm)



        #Data prep     
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = './datasets/'+args.dataset+'/'

        dset_test = NBADataset(
                data_set+'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True)

        loader_test = DataLoader(
                dset_test,
                batch_size=1,#This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=0,
                collate_fn=my_collate
        )



        #Defining the model 
        model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=5,pred_seq_len=args.pred_seq_len).cuda()
        model.load_state_dict(torch.load(model_path))


        loss_ =999999
        print("Testing ....")
        loss,result,result_A,names,ids= test()
        loss_= min(loss_,loss)
        loss_ls.append(loss_)
        print("MSELoss:",loss)

        features_dir = "./datasets/match/ba/"
        all_feature_files = os.listdir(features_dir)  # 列出数据文件夹中的所有文件
        all_feature_files = [os.path.join(features_dir, _path) for _path in all_feature_files]  # 获得所有文件的完整路径
        data = pd.read_csv(all_feature_files[-1])
        data = data.drop(['Unnamed: 0', 'GAME_DATE', 'SEASON_ID', 'HOA', 'ENEMY', 'RATIO'], axis=1)
        data = data.values.tolist()

        with open('result.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['ID','Game', 'Time', 'Team','TEAM','ENEMY','FGM', 'FG3M', 'FTM', 'OREB',
                       'DREB', 'AST', 'TOV', 'PF','PTS']
            writer.writerow(headers)
            #62
            for i, game in enumerate(result):
                #5
                for t in range(len(game)):
                    #30
                    for j in range(len(game[t])):
                        row = [ids[i][j][t]] +\
                              [i + 1, t + 1, j + 1,names[i][t][j][0], names[i][t][j][1],"{:.2f}".format(float(game[t][j][0])), "{:.2f}".format(float(game[t][j][1]))] + \
                              [f"{float(val):.2f}" for val in game[t][j][2:].flatten()]+\
                              ["{:.2f}".format(float(game[t][j][1])+float(game[t][j][2])+2*float(game[t][j][0]))]

                        writer.writerow(row)

        # with open('result_A.csv',mode='w',newline='') as file:
        #     writer = csv.writer(file)
        #     for i, game in enumerate(result_A):
        #         for t in range(len(game)):
        #             for j in range(len(game[t])):
        #                 row = [f"{float(val):.2f}" for val in game[t][j][:].flatten()]
        #                 writer.writerow(row)
    print("*"*50)

    print("Avg MSELoss:",sum(loss_ls)/len(paths))
