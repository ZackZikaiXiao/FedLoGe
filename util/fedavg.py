# python version 3.7.1
# -*- coding: utf-8 -*-

import copy
from pydoc import cli
import torch
import numpy as np
from torch import nn

# Averaging processing in coordinator for various model weights

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        #print('k',k)
        for i in range(1, len(w)):
            #print('i',i)
            w_avg[k] += w[i][k]
            #print(w[i][k])
        #w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = w_avg[k] / len(w)
    return w_avg


def diver_cal(model_flag, w_g, w_l):
    w_flag = model_flag.state_dict()
    for k in w_flag.keys():
        #print(k)
        w_flag[k] = w_g[k] - w_l[k]
    model_flag.load_state_dict(w_flag)
    sum_diver = 0
    for param in model_flag.parameters():
        #print('param',param)
        se = torch.sum(param**2)
        #print('se', se)
        sum_diver += se.detach().cpu().numpy()

    #sum_diver = s
    #print(s.detach().numpy())
    return sum_diver

'''
def diver_cal(w_g, w_l):
    w_flag = copy.deepcopy(w_g)
    sum_diver = 0
    for k in w_flag.keys():
        diff = sum((w_g[k] - w_l[k])**2)
        se = sum(diff).cpu().numpy()
        sum_diver += se
    
    return sum_diver
'''

def FedAvg_noniid_classifier(w, dict_len):
    model = copy.deepcopy(w[0])
    for i in range(len(w)):
        w[i] = w[i].state_dict()
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
            #w_avg[k] += w[i][k]
        #w_avg[k] = w_avg[k] / len(w)
        w_avg[k] = w_avg[k] / sum(dict_len)
    model.load_state_dict(w_avg)
    return model

# 针对于classifier的norm aggregation
def cls_norm_agg(w, dict_len, l_heads, distributions):
    

    model = copy.deepcopy(w[0])
    for i in range(len(w)):
        w[i] = w[i].state_dict()
    
    # 新建一个全空的return tensor
    w_avg = copy.deepcopy(w[0])
    w_avg = {k: torch.zeros_like(v) for k, v in w_avg.items()}


    # norm_map[i][c]: 第i的client，类别为c的vector的norm
    norm_map = []
    for i in range(len(w)):
        # 权重norm初始化
        norm = torch.norm(l_heads[i].weight, p=2, dim=1)
        # 将g_head.weight转换为torch.nn.Parameter类型
        norm_map.append(norm)
    # 将norm_map从list转换为tensor
    norm_map = torch.stack(norm_map)

    


    distributions = torch.from_numpy(distributions)
    distributions = distributions.to(norm_map.dtype)
    classes = l_heads[0].out_features


    weight_map = copy.deepcopy(distributions)
    for i in range(0, len(w)):
        for c in range(0, classes):
           weight_map[i][c] =  (dict_len[i] / sum(dict_len)) * (distributions[i][c] / torch.sum(distributions, dim=0)[c])
        #    weight_map[i][c] =  (dict_len[i] / sum(dict_len)) * (norm_map[i][c] / torch.sum(norm_map, dim=0)[c])


    # weight
    for i in range(0, len(w)):  # 对于每个client
        for c in range(0, classes):    # 对于每个类别
            # norm
            # w_avg['weight'][c] += w[i]['weight'][c] * (dict_len[i] / sum(dict_len)) * (norm_map[i][c] / torch.sum(norm_map, dim=0)[c])  # 第一个client的第一个类别向量
            # w_avg['weight'][c] += w[i]['weight'][c] * (norm_map[i][c] / torch.sum(norm_map, dim=0)[c]) 

            # dataset size
            # w_avg['weight'][c] += w[i]['weight'][c] * (dict_len[i] / sum(dict_len)) 

            # distritbution
            # w_avg['weight'][c] += w[i]['weight'][c] * (dict_len[i] / sum(dict_len)) * (distributions[i][c] / torch.sum(distributions, dim=0)[c])  
            
            # 归一化加权dataset size和distribution
            w_avg['weight'][c] += w[i]['weight'][c] * (weight_map[i][c] / torch.sum(weight_map, dim=0)[c])  



    # bias
    for i in range(0, len(w)):  # 对于每个client
        for c in range(0, classes):
            # w_avg['bias'][c] += w[i]['bias'][c] * (dict_len[i] / sum(dict_len)) * (norm_map[i][c] / torch.sum(norm_map, dim=0)[c])
            # w_avg['bias'][c] += w[i]['bias'][c] * (norm_map[i][c] / torch.sum(norm_map, dim=0)[c])

            # w_avg['bias'][c] += w[i]['bias'][c] * (dict_len[i] / sum(dict_len)) 
            # w_avg['bias'][c] += w[i]['bias'][c] * (dict_len[i] / sum(dict_len)) * (distributions[i][c] / torch.sum(distributions, dim=0)[c])
            
            # 归一化加权dataset size和distribution
            w_avg['bias'][c] += w[i]['bias'][c]  * (weight_map[i][c] / torch.sum(weight_map, dim=0)[c])



    model.load_state_dict(w_avg)
    return model

def aggregate_scalers(scalars, dict_len):
    # w_avg = copy.deepcopy(g_heads[0])
    # g_head = [g_heads[i].weight.data * dict_len[i] for i in range(len(dict_len))]
    # stacked_g_head = torch.stack(g_head, dim=0)
    # sum_g_head = torch.sum(stacked_g_head, dim=0)

    # g_head_bias = [g_heads[i].bias.data * dict_len[i] for i in range(len(dict_len))]
    # stacked_bias = torch.stack(g_head_bias, dim=0)
    # sum_bias = torch.sum(stacked_bias, dim=0)
    # w_avg.weight.data = sum_g_head/sum(dict_len)
    # w_avg.bias.data = sum_bias/sum(dict_len)
    # returned_linear = nn.Linear(512, 100).to(g_heads[0].device)
    # returned_linear.weight.data = torch.tensor(sum_g_head/sum(dict_len), device=scaler[0].device).clone().detach().requires_grad_(True)
    # returned_linear.bias.data = 

    scaler = [scalars[i].data * dict_len[i] for i in range(len(dict_len))]
    stacked_tensor = torch.stack(scaler, dim=0)
    sum_tensor = torch.sum(stacked_tensor, dim=0)
    return torch.tensor(sum_tensor/sum(dict_len), device=scaler[0].device).clone().detach().requires_grad_(True)
    # return torch.tensor(sum_tensor/sum(dict_len), device=scaler[0].device).clone().detach().requires_grad_(True)
    
    
    

def FedAvg_noniid(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
            #w_avg[k] += w[i][k]
        #w_avg[k] = w_avg[k] / len(w)
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg

def FedAvg_noniid_class_means(class_means_for_agg, dict_len):
    # 初始化一个字典，作为加权平均的结果
    aggregated_means = copy.deepcopy(class_means_for_agg[0])

    # 对每个类别的特征向量进行加权平均
    for k in aggregated_means.keys():
        aggregated_means[k] = aggregated_means[k] * dict_len[0]
        for i in range(1, len(class_means_for_agg)):
            if class_means_for_agg[i] is not None:
                aggregated_means[k] += class_means_for_agg[i][k] * dict_len[i]

        # 计算加权平均值
        aggregated_means[k] = aggregated_means[k] / sum(dict_len)

    return aggregated_means



def FedAvg_Rod(backbone_w_locals, linear_w_locals, dict_len):
    backbone_w_avg = FedAvg_noniid(backbone_w_locals, dict_len)
    linear_w_avg = FedAvg_noniid(linear_w_locals, dict_len)
    return backbone_w_avg, linear_w_avg

# 根据weight norm进行aggregation
# w: weights, dict_len:number of samples, beta: beta * weno + (1-beta) * avg
def weno_aggeration(w, dict_len, datasetObj, beta, round, start_round = 25):
    # fedavg的weight
    avg_w = copy.deepcopy(w[0]) 
    # 合并feature extractor(当然连同classifier一起合并了)
    for k in avg_w.keys():        
        avg_w[k] = avg_w[k] * dict_len[0] 
        for i in range(1, len(w)):
            avg_w[k] += w[i][k] * dict_len[i]
            #w_avg[k] += w[i][k]
        #w_avg[k] = w_avg[k] / len(w)
        avg_w[k] = avg_w[k] / sum(dict_len)
    # 计算weightnorm
    # wns_prop = []    # wns[i][j]:第i个client的classifier的第j类的client内占比（weight norms proportion）
    # for i in range(len(w)):
    #     wns_prop.append(torch.norm(w[i]["linear.weight"], p=2, dim=1) / torch.norm(w[i]["linear.weight"], p=2, dim=1).sum()) 

    # weno aggregation for classifier
    # weno_w["linear.weight"].zero_()
    # weno_w["linear.bias"].zero_()
    # class_wise_num = [0 for i in range(weno_w["linear.bias"].shape[0])]     # 每个类别的累计样本总数，十个类别则长度为十
    # for id_cls in range(weno_w["linear.bias"].shape[0]):   # 对于每一个类别而言    
    #     for id_client in range(len(w)):   # 对于每一个client
    #         weno_w["linear.weight"][id_cls] += w[id_client]["linear.weight"][id_cls] * wns_prop[id_client][id_cls] * dict_len[id_cls]
    #         weno_w["linear.bias"][id_cls] += w[id_client]["linear.bias"][id_cls] * wns_prop[id_client][id_cls] * dict_len[id_cls]
    #         class_wise_num[id_cls] += wns_prop[id_client][id_cls] * dict_len[id_cls]
    #     weno_w["linear.weight"][id_cls] / class_wise_num[id_cls]
    #     weno_w["linear.bias"][id_cls] / class_wise_num[id_cls]

    # 完全weno
    weno_classifier = copy.deepcopy(w[0])
    client_distribution = datasetObj.training_set_distribution
    
    # 按照sample proportion进行聚合
    client_distribution = client_distribution.astype(np.float64)
    for i in range(len(client_distribution)):
        client_distribution[i] /= sum(client_distribution[i])
    weno_classifier["linear.weight"].zero_()
    weno_classifier["linear.bias"].zero_()
    class_wise_num = [0 for i in range(weno_classifier["linear.bias"].shape[0])]     # 每个类别的累计样本总数，十个类别则长度为十
    for id_cls in range(weno_classifier["linear.bias"].shape[0]):   # 对于每一个类别而言    
        for id_client in range(len(w)):   # 对于每一个client
            weno_classifier["linear.weight"][id_cls] += w[id_client]["linear.weight"][id_cls] * client_distribution[id_client][id_cls] * dict_len[id_cls]
            weno_classifier["linear.bias"][id_cls] += w[id_client]["linear.bias"][id_cls] * client_distribution[id_client][id_cls] * dict_len[id_cls]
            class_wise_num[id_cls] += client_distribution[id_client][id_cls] * dict_len[id_cls]
        weno_classifier["linear.weight"][id_cls] / class_wise_num[id_cls]
        weno_classifier["linear.bias"][id_cls] / class_wise_num[id_cls]

    # 加权
    if round > start_round:
        avg_w["linear.weight"] = beta * weno_classifier["linear.weight"] + (1 - beta) * avg_w["linear.weight"]
        avg_w["linear.bias"] = beta * weno_classifier["linear.bias"] + (1 - beta) * avg_w["linear.bias"]

    return avg_w

def Weighted_avg_f1(f1_list,dict_len):
    f1_avg = 0
    for i in range(len(dict_len)):
        f1_avg += f1_list[i]*dict_len[i]
    f1_avg = f1_avg/sum(dict_len)
    return f1_avg