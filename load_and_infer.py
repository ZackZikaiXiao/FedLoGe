# python version 3.7.1
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import random
import torch

import pdb
import torch.nn as nn
from tqdm import tqdm
from options import args_parser, args_parser_cifar10
from util.update_baseline import LocalUpdate, globaltest, localtest, globaltest_calibra
from util.fedavg import *
# from util.util import add_noise
from util.dataset import *
from model.build_model import build_model
from util.dispatch import *
from util.losses import *

np.set_printoptions(threshold=np.inf)

dataset_switch = "cifar10"

def get_acc_file_path(args):

    rootpath = './temp/'
    if not os.path.exists(rootpath):  #for fedavg, beta = 0, 
        os.makedirs(rootpath)
 
    if args.balanced_global:
        rootpath+='global_' 
    rootpath += 'fl'
    if args.beta > 0: # set default mu = 1, and set beta = 1 when using fedprox
        #args.mu = 1
        rootpath += "_LP_%.2f" % (args.beta)
    fpath =  rootpath + '_acc_{}_{}_cons_frac{}_iid{}_iter{}_ep{}_lr{}_N{}_{}_seed{}_p{}_dirichlet{}_IF{}_Loss{}.txt'.format(
        args.dataset, args.model, args.frac, args.iid, args.rounds, args.local_ep, args.lr, args.num_users, args.num_classes, args.seed, args.non_iid_prob_class, args.alpha_dirichlet, args.IF, args.loss_type)
    return fpath
   
if __name__ == '__main__':
    # parse args
    if dataset_switch == 'cifar100':
        args = args_parser()
    elif dataset_switch == 'cifar10':
        args = args_parser_cifar10()
    # print("STOP")
    # return
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    fpath = get_acc_file_path(args)
    f_acc = open(fpath,'a')
    print(fpath)

    # pdb.set_trace()

    # myDataset containing details and configs about dataset(note: details)
    datasetObj = myDataset(args)
    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_balanced_dataset(datasetObj.get_args())  # CIFAR10
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_imbalanced_dataset(datasetObj.get_args())  # IMBALANCEDCIFAR10
        # data_path = './cifar_lt/'
        # trans_val = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])],
        # )
        # dataset_test_lt = IMBALANCECIFAR10(data_path, imb_factor=args.IF,train=False, download=True, transform=trans_val)
    
    
    print(len(dict_users))
    # pdb.set_trace()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # build model
    model = build_model(args) 
    # acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(netglob).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[0], user_id = 0)
    # copy weights
    w_glob = model.state_dict()  # return a dictionary containing a whole state of the module
    # w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]
    # w_locals = fedbn_assign(w_locals, w_glob)
    # w_locals = dispatch_fedper(w_locals, w_glob)

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

    # acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), g_head, dataset_test, args, dataset_class = datasetObj)

    # add fl training
    load_dir = "./output/ours/d/"
    # save_id = "73"
    model = torch.load(load_dir + "model_499.pth").to(args.device)
    g_head = torch.load(load_dir + "g_head_499.pth").to(args.device)
    g_aux = torch.load(load_dir + "g_aux_499.pth").to(args.device)
    l_heads = []
    for i in range(args.num_users):
        l_heads.append(torch.load(load_dir +  "l_head_" + str(i) + ".pth").to(args.device))

    # norm = torch.norm(g_aux.weight, p=2, dim=1)
    # # 将g_head.weight转换为torch.nn.Parameter类型
    # g_aux.weight = nn.Parameter(g_aux.weight / norm.unsqueeze(1))
    # g_head.weight = nn.Parameter(torch.ones_like(g_head.weight))

    # print("l_head", torch.norm(l_heads[0].weight, p=2, dim=1))
    # print("g_head", torch.norm(g_head.weight, p=2, dim=1))
    # print("g_aux", torch.norm(g_aux.weight, p=2, dim=1))
    


    # local tuning
    w_glob = model.state_dict()  # return a dictionary containing a whole state of the module
    w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]
    g_auxs_intervaria = []
    epoch = 5
    for client_id in range(args.num_users):  # training over the subset, in fedper, all clients train
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[client_id])
        w_locals[client_id], g_aux_intervaria, l_heads[client_id], loss_local = local.update_weights_norm_init(net=copy.deepcopy(model).to(args.device), g_head = copy.deepcopy(g_head).to(args.device), g_aux = copy.deepcopy(g_aux).to(args.device), l_head = copy.deepcopy(l_heads[client_id]).to(args.device), seed=args.seed, net_glob=model.to(args.device), epoch=epoch)
        g_auxs_intervaria.append(g_aux_intervaria)


    # global tuning
    




    # global test
    # 将g_head除以norm
    # g_head.weight = g_head.weight / norm.unsqueeze(1)
    acc_s2, global_3shot_acc = globaltest_calibra(copy.deepcopy(model).to(args.device), g_aux = copy.deepcopy(g_aux).to(args.device), test_dataset = dataset_test, args = args, dataset_class = datasetObj)

    # acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_head).to(args.device), dataset_test, args, dataset_class = datasetObj)
    print(acc_s2)
    print(global_3shot_acc)



     # local test 
    acc_list = []
    f1_macro_list = []
    f1_weighted_list = []
    acc_3shot_local_list = []
    for i in range(args.num_users):
        # model.load_state_dict(copy.deepcopy(w_locals[i]))
        # print('copy sucess')
        acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_auxs_intervaria[i]).to(args.device), copy.deepcopy(l_heads[i]).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[i], user_id = i)
        # print('local test success')
        acc_list.append(acc_local)
        f1_macro_list.append(f1_macro)
        f1_weighted_list.append(f1_weighted)
        acc_3shot_local_list.append(acc_3shot_local) 

    # start:calculate acc_3shot_local
    avg3shot_acc={"head":0, "middle":0, "tail":0}
    divisor = {"head":0, "middle":0, "tail":0}
    for i in range(len(acc_3shot_local_list)):
        avg3shot_acc["head"] += acc_3shot_local_list[i]["head"][0]
        avg3shot_acc["middle"] += acc_3shot_local_list[i]["middle"][0]
        avg3shot_acc["tail"] += acc_3shot_local_list[i]["tail"][0]
        divisor["head"] += acc_3shot_local_list[i]["head"][1]
        divisor["middle"] += acc_3shot_local_list[i]["middle"][1]
        divisor["tail"] += acc_3shot_local_list[i]["tail"][1]
    avg3shot_acc["head"] /= divisor["head"]
    avg3shot_acc["middle"] /= divisor["middle"]
    avg3shot_acc["tail"] /= divisor["tail"]
    # end 
    
    # start: calculate 3shot of each client
    # # three_shot_client = [{"head":0, "middle":0, "tail":0} for i in range(len(acc_3shot_local_list))]
    for i in range(len(acc_3shot_local_list)):
        acclist = []
        if acc_3shot_local_list[i]["head"][1] == True:
            acclist.append(acc_3shot_local_list[i]["head"][0])
        else:
            acclist.append(0)

        if acc_3shot_local_list[i]["middle"][1] == True:
            acclist.append(acc_3shot_local_list[i]["middle"][0])
        else:
            acclist.append(0)
            
        if acc_3shot_local_list[i]["tail"][1] == True:
            acclist.append(acc_3shot_local_list[i]["tail"][0])
        else:
            acclist.append(0)
        print("3shot of client {}:head:{}, middle:{}, tail{}".format(i, acclist[0], acclist[1], acclist[2]))
    # end

    avg_local_acc = sum(acc_list)/len(acc_list)
    # print('Calculate the local average acc')

    idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
    dict_len = [len(dict_users[idx]) for idx in idxs_users]
    avg_f1_macro = Weighted_avg_f1(f1_macro_list,dict_len=dict_len)
    avg_f1_weighted = Weighted_avg_f1(f1_weighted_list,dict_len)

    rnd = 0
    print('round %d, local average test acc  %.4f \n'%(rnd, avg_local_acc))
    print('round %d, local macro average F1 score  %.4f \n'%(rnd, avg_f1_macro))
    print('round %d, local weighted average F1 score  %.4f \n'%(rnd, avg_f1_weighted))
    print('round %d, global test acc  %.4f \n'%(rnd, acc_s2))
    print('round %d, average 3shot acc: [head: %.4f, middle: %.4f, tail: %.4f] \n'%(rnd, avg3shot_acc["head"], avg3shot_acc["middle"], avg3shot_acc["tail"]))
    print('round %d, global 3shot acc: [head: %.4f, middle: %.4f, tail: %.4f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))

    print("l_head", torch.norm(l_heads[0].weight, p=2, dim=1))
    print("g_head", torch.norm(g_head.weight, p=2, dim=1))
    print("g_aux", torch.norm(g_aux.weight, p=2, dim=1))
    # for i in range(args.num_users):
    #         torch.save(l_heads[i], "./output/aux_ghead_lhead/" + "l_head_" + str(i) + ".pth")