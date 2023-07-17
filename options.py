# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training in fine_tuning stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs in preprocessing stage")    # 5
    parser.add_argument('--frac', type=float, default=0.2, help="fration of selected clients in preprocessing stage")

    parser.add_argument('--num_users', type=int, default=20, help="number of uses: K")      # 40   
    parser.add_argument('--local_bs', type=int, default=8, help="local batch size: B")    
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")         # 0.03 
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")    
    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal, 0 for fedavg, 0.01 for fedprox, 5 for noise fl")

    # other arguments
    # parser.add_argument('--server', type=str, default='none', help="type of server")
    parser.add_argument('--model', type=str, default='resnet34', help="model name")     # 18
    parser.add_argument('--dataset', type=str, default='cifar100', help="name of dataset")   # cifar 10
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=1, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=0.5)  
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")   # 10
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--seed', type=int, default=3407, help="random seed, default: 1")

    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')

    parser.add_argument('--balanced_global', default=False, action='store_true', help="balanced global distribution or long tailed global distribution, clients are heterogeneous.")
    parser.add_argument('--IF', type=float, default=0.01, help="imbalance factor: Min/Max") # 0.1
    parser.add_argument('--gpu', type=int, default=7, help="gpu")
    return parser.parse_args()


def args_parser_cifar10():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training in fine_tuning stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs in preprocessing stage")    # 5
    parser.add_argument('--frac', type=float, default=0.2, help="fration of selected clients in preprocessing stage")

    parser.add_argument('--num_users', type=int, default=40, help="number of uses: K")      # 40   
    parser.add_argument('--local_bs', type=int, default=8, help="local batch size: B")    
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")         # 0.03 
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")    
    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal, 0 for fedavg, 0.01 for fedprox, 5 for noise fl")

    # other arguments
    # parser.add_argument('--server', type=str, default='none', help="type of server")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")     # 18
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")   # cifar 10
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=1, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=0.5)  
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")   # 10
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--seed', type=int, default=3407, help="random seed, default: 1")

    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')

    parser.add_argument('--balanced_global', default=False, action='store_true', help="balanced global distribution or long tailed global distribution, clients are heterogeneous.")
    parser.add_argument('--IF', type=float, default=0.01, help="imbalance factor: Min/Max") # 0.1
    parser.add_argument('--gpu', type=int, default=7, help="gpu")
    return parser.parse_args()