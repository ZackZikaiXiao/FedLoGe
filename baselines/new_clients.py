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
from util.update_baseline import LocalUpdate, globaltest, localtest, globaltest_classmean, globaltest_calibra, globaltest_feat_collapse
from util.fedavg import *
# from util.util import add_noise
from util.dataset import *
from model.build_model import build_model
from util.dispatch import *
from util.losses import *
from util.etf_methods import ETF_Classifier
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=np.inf)

load_switch = True  # True / False
save_switch = False # True / False
cls_switch = "ETF" # ETF / sparfix / dropout_ETF / w_dropout_ETF / PR_ETF
pretrain_cls = False
dataset_switch = 'cifar100' # cifar10 / cifar100
aggregation_switch = 'fedavg' # fedavg / class_wise
global_test_head = 'g_aux'  # g_aux / g_head
internal_frozen = False  # True / False
loss_switch = "None" # focous_loss / any others


    
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
         
    print(len(dict_users))
    # pdb.set_trace()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # build model
    model = build_model(args) 
    # 冻结特定层
    if internal_frozen:
        model.layer1[0].conv1.weight.requires_grad = False

        # 如果Conv2d层有bias，也需要冻结
        if model.layer1[0].conv1.bias is not None:
            model.layer1[0].conv1.bias.requires_grad = False
    # acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(netglob).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[0], user_id = 0)
    # copy weights
    w_glob = model.state_dict()  # return a dictionary containing a whole state of the module
    w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]
    # w_locals = fedbn_assign(w_locals, w_glob)
    # w_locals = dispatch_fedper(w_locals, w_glob)

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

    in_features = model.linear.in_features
    out_features = model.linear.out_features


    g_aux = nn.Linear(in_features, out_features).to(args.device)
    
    l_heads = []
    for i in range(args.num_users):
        l_heads.append(nn.Linear(in_features, out_features).to(args.device))

    if load_switch == True:
        rnd = 499
        load_dir = "./cifar100_100_0_05_sparse_etf/" # output1  output_nospar
        model = torch.load(load_dir + "model_" + str(rnd) + ".pth").to(args.device)
        model_origin = copy.deepcopy(model)
        g_head = torch.load(load_dir + "g_head_" + str(rnd) + ".pth").to(args.device)
        g_aux = torch.load(load_dir + "g_aux_" + str(rnd) + ".pth").to(args.device)
        for i in range(args.num_users):
            l_heads[i] = torch.load(load_dir + "l_head_" + str(i) + ".pth").to(args.device)
        w_glob = model.state_dict()  # return a dictionary containing a whole state of the module
            # w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]


    # GA-LR
    GA_LR = True
    # GA_LR = False
    if GA_LR:
        cali_alpha = torch.norm(g_aux.weight, dim=1)
        cali_alpha = torch.pow(cali_alpha, 1)
        inverse_cali_alpha = 1.7 / cali_alpha
        inverse_cali_alpha = inverse_cali_alpha.view(-1, 1)
        g_aux.weight = torch.nn.Parameter(g_aux.weight * inverse_cali_alpha)

    # acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_head).to(args.device), copy.deepcopy(g_aux).to(args.device), dataset_test, args, dataset_class = datasetObj)
    # add fl training
    for rnd in tqdm(range(args.rounds)):
        # if rnd % 1 == 0:
        #     g_head.reassign()
        g_auxs = []
        w_locals = []
        # w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)

        ## local training       
        g_head.train()  # 开启dropout
        for client_id in idxs_users:  # training over the subset, in fedper, all clients train
            # model.load_state_dict(copy.deepcopy(w_locals[client_id]))
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[client_id])
            w_local, g_aux_temp, l_heads[client_id], loss_local = local.update_weights_new_client(net=copy.deepcopy(model).to(args.device), g_head = copy.deepcopy(g_head).to(args.device), g_aux = copy.deepcopy(g_aux).to(args.device), l_head = l_heads[client_id], epoch=args.local_ep, loss_switch = loss_switch)
            g_auxs.append(g_aux_temp)
            w_locals.append(w_local)
        g_head.eval()  # 关闭dropout
        ## aggregation 
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg_noniid(w_locals, dict_len)

        # if aggregation_switch == 'fedavg':
        #     g_aux = FedAvg_noniid_classifier(g_auxs, dict_len)
        # elif aggregation_switch == 'class_wise':
        #     g_aux = cls_norm_agg(g_auxs, dict_len, l_heads=l_heads, distributions = datasetObj.training_set_distribution)


        ## assign
        w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]
        # w_locals = dispatch_fedper(w_locals, w_glob)

        ## global test
        model.load_state_dict(copy.deepcopy(w_glob))

        if global_test_head == 'g_head':
            acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_head).to(args.device), dataset_test, args, dataset_class = datasetObj)
        elif global_test_head == 'g_aux':
            acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_aux).to(args.device), dataset_test, args, dataset_class = datasetObj)


        # local test 
        acc_list = []
        f1_macro_list = []
        f1_weighted_list = []
        acc_3shot_local_list = []       #####################
        for i in range(args.num_users):
            # model.load_state_dict(copy.deepcopy(w_locals[i]))
            # print('copy sucess')
            acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(model_origin).to(args.device), copy.deepcopy(g_auxs[i]).to(args.device), copy.deepcopy(l_heads[i]).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[i], user_id = i)
            # print('local test success')
            acc_list.append(acc_local)
            f1_macro_list.append(f1_macro)
            f1_weighted_list.append(f1_weighted)
            acc_3shot_local_list.append(acc_3shot_local) ###################


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
        
        avg_f1_macro = Weighted_avg_f1(f1_macro_list,dict_len=dict_len)
        avg_f1_weighted = Weighted_avg_f1(f1_weighted_list,dict_len)

        print('round %d, local average test acc  %.4f \n'%(rnd, avg_local_acc))
        print('round %d, local macro average F1 score  %.4f \n'%(rnd, avg_f1_macro))
        print('round %d, local weighted average F1 score  %.4f \n'%(rnd, avg_f1_weighted))
        print('round %d, global test acc  %.4f \n'%(rnd, acc_s2))
        print('round %d, average 3shot acc: [head: %.4f, middle: %.4f, tail: %.4f] \n'%(rnd, avg3shot_acc["head"], avg3shot_acc["middle"], avg3shot_acc["tail"]))
        print('round %d, global 3shot acc: [head: %.4f, middle: %.4f, tail: %.4f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
        
        print("l_head", torch.norm(l_heads[0].weight, p=2, dim=1))
        # print("g_head", torch.norm(g_head.weight, p=2, dim=1))
        print("g_aux", torch.norm(g_aux.weight, p=2, dim=1))
        break   # 结束
    torch.cuda.empty_cache()
