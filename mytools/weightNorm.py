# 探索classifier的weight norm和class number之间的关系

from http import client
import os
import copy
import numpy as np
import random
import torch
import pdb
from tqdm import tqdm
import math
from matplotlib import pyplot as plt
# 原始分布倒置过来；类别数量特别多的时候；cifar100；降低异质性难度(p alpha)，client降数量（10个），每个client数据多一点；看global的l2 norm；独立训练
# 不用覆盖的方式
# experi2.单次的梯度反应类别分布 expe3. 累计，v2中的图复现
client_distrition = [[ 89,   0,  12,  15,  59,  20,   2,  36,  86,   5],
 [ 44,  22,   7,  31,   3,  16,   7,  12,   5,  13],
 [  0,   8,  34,  13,  41,  20,  20,  11,  59,  28],
 [ 33,  17,  54,   2,  19,  21,  44,  38,   0,  59],
 [ 21,  23,  29,  23,  35,  31,  18,   3,   4,  22],
 [ 86,   4,   1,  73,   8,  16,  32,  38,  47,  17],
 [  9,   7,   6,  60,  44,  65,   3,  12,  47,  48],
 [  9,  20,   7,   5,  67,  31,  65,  15,  12,   1],
 [  4,   7,  11,  20,  21,  17,  63,  19,   2,  54],
 [ 16,   5,  58,  22,  31,  16,   4,  41,  27,  46],
 [  4,   8,  47,  17,  50,   9,  23,   4,  10,  11],
 [ 28,  49,  30,   2,  12,  17,  23,  98,  23,  24],
 [ 43,  44,  12,  16,   9,   7,  22,  15,  22,   9],
 [ 34,  15,   1,  13,   6,   0,  55,  22,  42,  38],
 [ 39,   1,   2,  49,  14,  24,   5,  54,   4,   4],
 [ 10,  40,  39,  54,  54,  55,   2,   4,  24,  67],
 [ 55,   4,  39,  55,  12,  30,   0,  57,  61,   5],
 [ 18,  23,   3,   3,   3,   3,   3,  27,  33,   1],
 [ 52,  31,   3,   4,   4,   9,  30,  26, 100,  39],
 [ 30,  68,   8,   7,   4,  37,   6,  18,   4,   4],
 [  2,   2,  42,  16,  41,  54,   4,  29,  23,  29],
 [  2,  16,   5,  14,  67,   5,  57,  43,  25,   8],
 [  0,  40,   1,  54,  21,  67,   8,  13,  73,  72],
 [  1,  17,  50,  33,  11,  36,   2,  18,   5,   5],
 [ 18,  31, 120,  35,  72,  41,  57,  19,  30,  11],
 [  4,  30,  16,  41,  24,   9,  54,  39,   7,   0],
 [  0,  44,  31,  25,  10,   0,  13,  60,   1,  15],
 [ 41,  23,   8,  41,   2,   0,  22,  39,   5,   0],
 [ 14,  42,  16,  23,   3,  19,  30,  14,  27,  41],
 [ 38,  26,  19,  32,  14,  21,  21,  35,  32,  20],
 [ 18,  65,  10,  22,   7,   6,  47,   3,  49,  18],
 [ 17,  18,  47,  14,  17,  84,   5,   9,   9,  69],
 [ 18,  44,  79,   5,  21,  24,   4,   3,   2,   5],
 [ 19,  19,  22,   2,  60,   6,  13,  16,  11,  10],
 [ 44,   5,  18,   3,  15,  40,  13,   6,  18,   0],
 [ 18,  36,  16,  22,   4,   1,   6,   9,  10,  30],
 [ 66,  13,  44,   1,  43,  23,   0,   8,  18,  35],
 [  0,   3,  18,   3,  33,  10,   4,  14,   2,  34],
 [ 40,  71,  16,  70,   3,   5,   2,  24,  11,   0],
 [  3,  38,   4,  46,  21,  91, 194,  35,  12,  87]]
 
def l2_distance(list1, list2):
    assert len(list1) == len(list2)
    l2_distance = 0
    for i in range(len(list1)):
        l2_distance += math.pow(list1[i] - list2[i], 2)
    return math.sqrt(l2_distance)

def MaxMinNormalization(list):
    for i in range(len(list)):
        list[i] = (list[i] - min(list)) / (max(list) - min(list))
    return list

def multiple(list, item):
    for i in range(len(list)):
        list[i] *= item
    return list

def normalization(list):
    list = np.array(list)
    list = (list - list.mean()) / list.std()
    list = list.tolist()
    return list


def cal_weight_norm(weight):
    sum = 0
    para_num = 0
    for k in weight.keys():
        if k == "linear.weight":
            linear_weight = weight[k]
            weight_norm = torch.norm(linear_weight, p=2, dim=1).tolist()
        # print(k)
    return weight_norm
        # print(weight[k])

def visualization(weight_norm, distribution, id, save_dir = "./visualization/"):
    weight_norm = normalization(weight_norm)
    distribution = normalization(distribution)
    # 排个序哦，从大到小
    for i in range(len(distribution)):
        for j in range(0,len(distribution) - i - 1):
            if distribution[j] < distribution[j+1]:
                distribution[j], distribution[j+1] = distribution[j+1], distribution[j]
                weight_norm[j], weight_norm[j+1] = weight_norm[j+1], weight_norm[j]
    fig = plt.figure(dpi = 500)
    x = range(len(weight_norm))
    plt.plot(x, weight_norm, label='weight_norm')
    plt.plot(x, distribution, label="distribution")

    plt.savefig(save_dir + str(id))

if __name__ == "__main__":
    # model = torch.load("../output/netglob.pth")
    local_weights = []
    client_nums = 40
    for client_id in range(client_nums):
        local_weights.append(torch.load("../output/" + "w_local_" + str(client_id) + ".pth"))
        weight_norm = cal_weight_norm(local_weights[-1])
        print(weight_norm)
        visualization(weight_norm, client_distrition[client_id], client_id, save_dir = "./visualization/")

    print("end")