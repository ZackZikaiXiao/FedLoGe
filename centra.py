# python version 3.7.1
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import random
import torch

import pdb

from tqdm import tqdm
from options import args_parser
from util.update_baseline import *
from util.fedavg import *
# from util.util import add_noise
from util.dataset import *
from model.build_model import build_model
from util.dispatch import *

np.set_printoptions(threshold=np.inf)



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
    args = args_parser()
    # print("STOP")
    # return
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

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
    # args.num_classes = 200
    model = build_model(args) 

    # model = torch.load("/home/zikaixiao/zikai/aapfl/fl_gba_cifar100/output/fedbabu_spar06/netglob_499.pth")
    # acc_s2, global_3shot_acc = globaltest_villina(copy.deepcopy(model).to(args.device), dataset_test, args, dataset_class = datasetObj)
    # print(acc_s2)
    # print(global_3shot_acc)

    # acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(netglob).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[0], user_id = 0)
    # copy weights
    w_glob = model.state_dict()  # return a dictionary containing a whole state of the module
    w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]
    # w_locals = fedbn_assign(w_locals, w_glob)
    w_locals = dispatch_fedper(w_locals, w_glob)

    # model = torch.load("/home/zikaixiao/zikai/aapfl/fl_gba_cifar100/output/fedbabu_spar06/netglob_499.pth")
    # acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), dataset_test, args, dataset_class = datasetObj)
    # print(acc_s2)
    # print(global_3shot_acc)
    
    
    
    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

   
    # add fl training
    for rnd in tqdm(range(args.rounds)):
        # w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
                
        ## local training       
        client_id = 0
        model.load_state_dict(copy.deepcopy(w_locals[client_id]))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[client_id])
        w_locals[client_id], loss_local = local.update_weights_backbone_only(net=copy.deepcopy(model).to(args.device), seed=args.seed, epoch=args.local_ep)
            # w_locals.append(copy.deepcopy(w_local))  # store every updated model
            # loss_locals.append(copy.deepcopy(loss_local))

        w_glob = w_locals[0]

        ## assign
        # w_locals = dispatch_fedper(w_locals, w_glob)

        ## global test
        model.load_state_dict(copy.deepcopy(w_glob))
        acc_s2, global_3shot_acc = globaltest_villina(copy.deepcopy(model).to(args.device), dataset_test, args, dataset_class = datasetObj)

      

   
        model.load_state_dict(copy.deepcopy(w_glob))

        print('round %d, global test acc  %.4f \n'%(rnd, acc_s2))
        print('round %d, global 3shot acc: [head: %.4f, middle: %.4f, tail: %.4f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
    torch.cuda.empty_cache()
