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

load_switch = False  # True / False
save_switch = False # True / False
cls_switch = "SSE-C" # SSE-C / sparfix / dropout_ETF / w_dropout_ETF / PR_ETF
pretrain_cls = False
dataset_switch = 'cifar100' # cifar10 / cifar100
aggregation_switch = 'fedavg' # fedavg / class_wise
global_test_head = 'g_head'  # g_aux / g_head
internal_frozen = False  # True / False
loss_switch = "None" # focous_loss / any others

class PRLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super(PRLinear, self).__init__(in_features, out_features, bias)
        self.eps = eps

    def forward(self, x):
        # compute the length of w and x. We find this is faster than the norm, although the later is simple.
        w_len = torch.sqrt((torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))).clamp_(min=self.eps))  # 1*num_classes
        x_len = torch.sqrt((x.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # batch*1

        # compute the cosine of theta and abs(sine) of theta.
        wx_len = torch.matmul(x_len, w_len).clamp_(min=self.eps)
        cos_theta = (torch.matmul(x, torch.t(self.weight)) / wx_len).clamp_(-1.0, 1.0)  # batch*num_classes
        abs_sin_theta = torch.sqrt(1.0 - cos_theta ** 2)  # batch*num_classes

        # PR Product
        out = wx_len * (abs_sin_theta.detach() * cos_theta + cos_theta.detach() * (1.0 - abs_sin_theta))

        # to save memory
        del w_len, x_len, wx_len, cos_theta, abs_sin_theta

        if self.bias is not None:
            out = out + self.bias

        return out


class dropout_ETF(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(dropout_ETF, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        etf = ETF_Classifier(in_features, out_features) 
        self.linear.weight.data = etf.ori_M.to(args.device)
        self.linear.weight.data = self.linear.weight.data.t()
        self.dropout_rate = dropout_rate
        self.mask = self.get_dropout_mask((in_features, in_features), dropout_rate)

    def forward(self, x):
        if self.training:  # apply dropout only during training
            x = x * self.mask[0:x.shape[0], :]  # Change matrix multiplication to element-wise multiplication
        x = self.linear(x)
        return x
    
    def reassign(self):
        with torch.no_grad():
            self.mask = self.get_dropout_mask((in_features, self.linear.in_features), self.dropout_rate)

    def get_dropout_mask(self, shape, dropout_rate):
        # save current RNG state
        rng_state = torch.random.get_rng_state()
        # generate dropout mask
        mask = (torch.rand(shape) > dropout_rate).float().to(args.device)
        # restore RNG state
        torch.random.set_rng_state(rng_state)
        return mask


# class dropout_ETF(nn.Module):
#     def __init__(self, in_features, out_features, dropout_rate):
#         super(dropout_ETF, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         etf = ETF_Classifier(in_features, out_features) 
#         self.linear.weight.data = etf.ori_M.to(args.device)
#         self.linear.weight.data = self.linear.weight.data.t()
#         self.dropout_rate = dropout_rate
#         self.mask = self.get_dropout_mask((1, in_features), dropout_rate)

#     def forward(self, x):
#         if self.training:  # apply dropout only during training
#             x = x * self.mask  # Change matrix multiplication to element-wise multiplication
#         x = self.linear(x)
#         return x
    
#     def reassign(self):
#         with torch.no_grad():
#             self.mask = self.get_dropout_mask((1, self.linear.in_features), self.dropout_rate)

#     def get_dropout_mask(self, shape, dropout_rate):
#         # save current RNG state
#         rng_state = torch.random.get_rng_state()
#         # generate dropout mask
#         mask = (torch.rand(shape) > dropout_rate).float().to(args.device)
#         # restore RNG state
#         torch.random.set_rng_state(rng_state)
#         return mask

class w_dropout_ETF(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(w_dropout_ETF, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        etf = ETF_Classifier(in_features, out_features) 
        self.linear.weight.data = etf.ori_M.to(args.device)
        self.linear.weight.data = self.linear.weight.data.t()
        self.dropout_rate = dropout_rate
        self.mask = self.get_dropout_mask((out_features, in_features), dropout_rate)

    def forward(self, x):
        if self.training:  # apply dropout only during training
            weight = self.linear.weight * self.mask  # Apply mask to the weight
        else:
            weight = self.linear.weight
        return F.linear(x, weight, self.linear.bias)  # Use F.linear to apply the modified weight
    
    def reassign(self):
        with torch.no_grad():
            self.mask = self.get_dropout_mask((self.linear.out_features, self.linear.in_features), self.dropout_rate)

    def get_dropout_mask(self, shape, dropout_rate):
        # save current RNG state
        rng_state = torch.random.get_rng_state()
        # generate dropout mask  
        mask = (torch.rand(shape) > dropout_rate).float().to(args.device)
        # restore RNG state
        torch.random.set_rng_state(rng_state)
        return mask
    
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
    # args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

    in_features = model.linear.in_features
    out_features = model.linear.out_features

    if cls_switch == "SSE-C":
        # 初始化ETF分类器 
        etf = ETF_Classifier(in_features, out_features) 
        # 新建线性层,权重使用ETF分类器的ori_M
        g_head = nn.Linear(in_features, out_features).to(args.device) 
        sparse_etf_mat = etf.gen_sparse_ETF(feat_in = in_features, num_classes = out_features, beta=0.6)

        etf_visual = False
        if etf_visual:
            # 设置全局字体
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 13


            # 假设sparse_etf_mat是您的向量矩阵
            # sparse_etf_mat = torch.rand((512, 100))

            # 计算模长
            magnitudes = torch.norm(sparse_etf_mat, dim=0).cpu().detach().numpy()

            # 初始化相似度矩阵
            cosine_similarity_matrix = torch.zeros((100, 100))

            # 计算余弦相似度
            for i in range(100):
                for j in range(100):
                    if i != j:
                        cosine_similarity_matrix[i, j] = torch.dot(sparse_etf_mat[:, i], sparse_etf_mat[:, j]) / (magnitudes[i] * magnitudes[j])
                    else:
                        cosine_similarity_matrix[i, j] = 1  # 向量与自身的相似度为1

            # 设置坐标轴刻度位置和标签

            # 使用seaborn的heatmap函数绘制热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(cosine_similarity_matrix.detach().numpy(), cmap="YlGnBu")
            
            xticks_positions = range(0, 100, 10)  # 横坐标每10个向量标记一次
            xticks_labels = map(str, range(0, 100, 10))
            plt.xticks(ticks=xticks_positions, labels=xticks_labels)

            yticks_positions = range(0, 100, 10)  # 纵坐标每10个向量标记一次
            yticks_labels = map(str, range(0, 100, 10))
            plt.yticks(ticks=yticks_positions, labels=yticks_labels)
            
            # plt.title('Cosine Similarity Between Vectors')
            plt.xlabel('Classes')
            plt.ylabel('Classes')

            # 保存图像
            plt.savefig("cosine_similarity_heatmap.pdf", dpi=500)

            plt.show()



        # g_head.weight.data = etf.ori_M.to(args.device)
        g_head.weight.data = sparse_etf_mat.to(args.device)
        g_head.weight.data = g_head.weight.data.t()
        

    elif cls_switch == "dropout_ETF":    
        g_head = dropout_ETF(in_features, out_features, dropout_rate=0.5).to(args.device)
    elif cls_switch == "w_dropout_ETF":    
        g_head = w_dropout_ETF(in_features, out_features, dropout_rate=0.5).to(args.device)
    elif cls_switch == "PR_ETF":
        etf = ETF_Classifier(in_features, out_features) 
        # 新建线性层,权重使用ETF分类器的ori_M
        g_head = PRLinear(in_features, out_features).to(args.device) 
        g_head.weight.data = etf.ori_M.to(args.device)
        g_head.weight.data = g_head.weight.data.t()
        nn.init.sparse_(g_head.weight, sparsity=0.6)
    elif cls_switch == "sparfix":
        g_head = nn.Linear(in_features, out_features).to(args.device)   # res34是512

        # kaiming初始化
        # torch.nn.init.kaiming_uniform_(g_head.weight, a=math.sqrt(5))
        # torch.nn.init.kaiming_normal_(g_head.weight)
        # # torch.nn.init.kaiming_normal_(g_head.bias)
        # # xavier初始化
        # torch.nn.init.xavier_uniform_(g_head.weight)
        # # torch.nn.init.xavier_uniform_(g_head.bias)
        # # 设置为0
        # torch.nn.init.constant_(g_head.weight, 0)
        # torch.nn.init.constant_(g_head.bias, 0)
        # # 设置为0.05
        # torch.nn.init.constant_(g_head.weight, 0.05)
        # torch.nn.init.constant_(g_head.bias, 0.5)
        # # 设置为0.1
        # torch.nn.init.constant_(g_head.weight, 0.1)
        # torch.nn.init.constant_(g_head.bias, 0.1)
        # # 均匀分布
        # torch.nn.init.uniform_(g_head.weight, a=0, b=1)
        # torch.nn.init.uniform_(g_head.bias, a=0, b=1)
        # # 高斯分布
        # torch.nn.init.normal_(g_head.weight, mean=0.0, std=0.5)
        # torch.nn.init.normal_(g_head.bias, mean=0.0, std=1.0)
        # # 正交分布
        # torch.nn.init.orthogonal_(g_head.weight, gain=1)
        # # torch.nn.init.orthogonal_(g_head.bias, gain=1)
        # # 稀疏初始化
        nn.init.sparse_(g_head.weight, sparsity=0.6)   # 在任意col上，10%类别为0


    if pretrain_cls == True:
        g_head.load_state_dict({k.replace('linear.', ''): v for k, v in torch.load("/home/zikaixiao/zikai/aapfl/pfed_lastest/demo.pth").items() if 'linear' in k})


    g_aux = nn.Linear(in_features, out_features).to(args.device)
    
    l_heads = []
    for i in range(args.num_users):
        l_heads.append(nn.Linear(in_features, out_features).to(args.device))

    if load_switch == True:
        rnd = 150
        load_dir = "./output/f/" # output1  output_nospar
        model = torch.load(load_dir + "model_" + str(rnd) + ".pth").to(args.device)
        # g_head = torch.load(load_dir + "g_head_" + str(rnd) + ".pth").to(args.device)
        # g_aux = torch.load(load_dir + "g_aux_" + str(rnd) + ".pth").to(args.device)
        # for i in range(args.num_users):
        #     l_heads[i] = torch.load(load_dir + "l_head_" + str(i) + ".pth").to(args.device)
        w_glob = model.state_dict()  # return a dictionary containing a whole state of the module
            # w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]

    # acc_s2, global_3shot_acc, g_head = globaltest_feat_collapse(copy.deepcopy(model).to(args.device), g_head, dataset_test, args, dataset_class = datasetObj)
    # globaltest_classmean
    
    # acc_s2, global_3shot_acc = globaltest_calibra(copy.deepcopy(model).to(args.device), copy.deepcopy(g_head).to(args.device), copy.deepcopy(g_aux).to(args.device), dataset_test, args, dataset_class = datasetObj)
    # add fl training
    # 初始化为1吧
    constant_scalar = torch.ones(1, requires_grad=True, device=args.device)
    # constant_scalar = constant_scalar.cuda()
    for rnd in tqdm(range(args.rounds)):
        # if rnd % 1 == 0:
        #     g_head.reassign()
        g_auxs = []
        w_locals = []
        constant_scalars = []
        # w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)

        ## local training       
        g_head.train()  # 开启dropout
        for client_id in idxs_users:  # training over the subset, in fedper, all clients train
            # model.load_state_dict(copy.deepcopy(w_locals[client_id]))
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[client_id])
            w_local, g_aux_temp, l_heads[client_id], loss_local, constant_scalar_tmp = local.update_weights_gaux(constant_scalar=copy.deepcopy(constant_scalar), net=copy.deepcopy(model).to(args.device), g_head = copy.deepcopy(g_head).to(args.device), g_aux = copy.deepcopy(g_aux).to(args.device), l_head = l_heads[client_id], epoch=args.local_ep, loss_switch = loss_switch)
            g_auxs.append(g_aux_temp)
            w_locals.append(w_local)
            constant_scalars.append(constant_scalar_tmp)
        
        g_head.eval()  # 关闭dropout
        ## aggregation 
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg_noniid(w_locals, dict_len)
        constant_scalar = aggregate_scalers(constant_scalars, dict_len)
        print("The value of constant scaler: ", constant_scalar)
        if aggregation_switch == 'fedavg':
            g_aux = FedAvg_noniid_classifier(g_auxs, dict_len)
        elif aggregation_switch == 'class_wise':
            g_aux = cls_norm_agg(g_auxs, dict_len, l_heads=l_heads, distributions = datasetObj.training_set_distribution)


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
            model.load_state_dict(copy.deepcopy(w_locals[i]))
            # print('copy sucess')
            acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_aux).to(args.device), copy.deepcopy(l_heads[i]).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[i], user_id = i)
            # print('local test success')
            acc_list.append(acc_local)
            f1_macro_list.append(f1_macro)
            f1_weighted_list.append(f1_weighted)
            acc_3shot_local_list.append(acc_3shot_local) ###################

        if save_switch == True:
            save_dir = "./output/40_30/"
            torch.save(model, save_dir + "model_" + str(rnd) + ".pth")
            torch.save(g_head, save_dir + "g_head_" + str(rnd) + ".pth")
            torch.save(g_aux, save_dir + "g_aux_" + str(rnd) + ".pth")
            for i in range(args.num_users):
                torch.save(l_heads[i], save_dir + "l_head_" + str(i) + ".pth")

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
    torch.cuda.empty_cache()
