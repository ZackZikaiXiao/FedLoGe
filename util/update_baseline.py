# python version 3.7.1
# -*- coding: utf-8 -*-

from cProfile import label
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from util.util import shot_split
import copy
from util.losses import *

import sklearn.metrics as metrics
from util.losses import FocalLoss



# 通过cient的id来划分local longtail的数据
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # self.loss_func = self.get_loss()  # loss function -- cross entropy
        # self.loss_func = nn.CrossEntropyLoss() # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def get_loss(self):
        if self.args.loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif self.args.loss_type == 'focal':
            return FocalLoss(gamma=1).cuda(self.args.gpu)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs),
                           batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_fedrep(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []

        # 先训练head，固定表征
        count = 0
        for p in net.parameters():
            if count >= 108:        # 108
                p.requires_grad = True
            else:
                p.requires_grad = False
            count += 1

        filter(lambda p: p.requires_grad, net.parameters())



        for iter in range(15):
            if iter == 10:
                # 再固定head，训练表征
                count = 0
                for p in net.parameters():
                    if count >= 108:        # 108
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                    count += 1

                filter(lambda p: p.requires_grad, net.parameters())

            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_gaux(self, net, g_head, g_aux, l_head, epoch, mu=1, lr=None, loss_switch=None):
        net.train()
        # train and update
        optimizer_g_backbone = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer_g_aux = torch.optim.SGD([
        #                     {'params': g_aux.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': g_aux.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 定义优化器
        # optimizer_l_head = torch.optim.SGD([
        #                     {'params': l_head.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': l_head.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()
        if loss_switch == "focus_loss":
            criterion_l = focus_loss(num_classes=100)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g_backbone.zero_grad()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()
                # net.zero_grad()

                # backbone
                features = net(images, latent_output=True)
                output_g_backbone = g_head(features)
                loss_g_backbone = criterion_g(output_g_backbone, labels)
                loss_g_backbone.backward()
                # max_grad = max(p.grad.data.abs().max() for p in net.parameters() if p.grad is not None)
                # print('Max gradient:', max_grad)
                optimizer_g_backbone.step()
                
                # g_aux
                output_g_aux = g_aux(features.detach())
                loss_g_aux = criterion_l(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g_backbone + loss_g_aux + loss_l_head
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)


    def update_weights_fedrod(self, net, g_head, g_aux, l_head, epoch, mu=1, lr=None):
        net.train()
        # train and update
        # optimizer_g_backbone = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(list(net.parameters()) + list(g_aux.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer_g_aux = torch.optim.SGD([
        #                     {'params': g_aux.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': g_aux.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l_head = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # 定义优化器
        # optimizer_l_head = torch.optim.SGD([
        #                     {'params': l_head.weight, 'weight_decay': 1e-1},  # 对权重使用weight_decay
        #                     {'params': l_head.bias, 'weight_decay': 0}  # 对偏置不使用weight_decay
        #                 ], lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g_aux.zero_grad()
                optimizer_l_head.zero_grad()

                # backbone + g_aux
                features = net(images, latent_output=True)
                output = g_aux(features)
                loss_g = criterion_g(output, labels)
                loss_g.backward()
                optimizer_g_aux.step()

                # p_head
                output_l_head = l_head(features.detach())
                loss_l_head = criterion_l(output_l_head, labels)
                loss_l_head.backward()
                optimizer_l_head.step()

                loss = loss_g + loss_l_head
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_balsoft_backup(self, net, g_head, g_aux, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update
        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g.zero_grad()
                optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                output_g = g_head(features)
                loss_g = criterion_g(output_g, labels)
                loss_g.backward()
                optimizer_g.step()

                # aux更新
                output_g_aux = g_aux(features.detach())
                loss_g_aux = criterion_g(output_g_aux, labels)
                loss_g_aux.backward()
                optimizer_g_aux.step()

                # p cls更新
                output_l = l_head(features.detach())
                loss_l = criterion_l(output_l, labels)
                loss_l.backward()
                optimizer_l.step()

                loss = loss_g + loss_g_aux + loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)
    

    def update_weights_phead_backup(self, net, g_aux, g_head, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update
        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # criterion_g = balsoft_loss
        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                optimizer_g.zero_grad()
                optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                # output_g = g_head(features)
                # loss_g = criterion_g(output_g, labels)
                # loss_g.backward()
                # optimizer_g.step()

                # aux更新
                # output_g_aux = g_aux(features.detach())
                # loss_g_aux = criterion_g(output_g_aux, labels)
                # loss_g_aux.backward()
                # optimizer_g_aux.step()

                # p cls更新
                output_l = l_head(features.detach())
                loss_l = criterion_l(output_l, labels)
                loss_l.backward()
                optimizer_l.step()

                loss = loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), g_aux, l_head, sum(epoch_loss) / len(epoch_loss)
    

    def update_weights_unlearning(self, net, g_aux, g_head, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update
        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(list(l_head.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(list(g_aux.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = torch.optim.SGD(list(g_aux.parameters()) + list(l_head.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        # criterion_g = balsoft_loss
        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()
        
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_ce = nn.CrossEntropyLoss()

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # optimizer_g.zero_grad()
                # optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()
                # optimizer.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                # output_g = g_head(features)
                # loss_g = criterion_g(output_g, labels)
                # loss_g.backward()
                # optimizer_g.step()

                # aux更新
                # output_g_aux = g_aux(features.detach())
                # loss_g_aux = criterion_g(output_g_aux, labels)
                # loss_g_aux.backward()
                # optimizer_g_aux.step()

                # 迁移学习
                outputs_g_aux = g_aux(features.detach())
                outputs_l_head = l_head(features.detach())
                
                outputs_g_aux_normalized = F.normalize(outputs_g_aux, dim=1)
                outputs_l_head_normalized = F.normalize(outputs_l_head, dim=1)

                
                # loss_ce = criterion_ce(outputs_g_aux, labels)
                loss_kl = criterion_kl(nn.functional.log_softmax(outputs_g_aux_normalized, dim=1),
                                    nn.functional.softmax(outputs_l_head_normalized, dim=1))
                loss = loss_kl
                loss.backward()
                optimizer_g_aux.step()


                # p cls更新
                # output_l = l_head(features.detach())
                # loss_l = criterion_l(output_l, labels)
                # loss_l.backward()
                # optimizer_l.step()

                # loss = loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), copy.deepcopy(g_aux), copy.deepcopy(l_head), sum(epoch_loss) / len(epoch_loss)
    

    def update_weights_norm_init(self, net, g_aux, g_head, l_head, seed, net_glob, epoch, mu=1, lr=None):

        net.train()
        # train and update


        # 权重norm初始化
        norm = torch.norm(l_head.weight, p=2, dim=1)
        # 将g_head.weight转换为torch.nn.Parameter类型
        g_aux.weight = nn.Parameter(g_aux.weight * norm.unsqueeze(1))


        optimizer_g = torch.optim.SGD(list(net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_l = torch.optim.SGD(l_head.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer_g_aux = torch.optim.SGD(g_aux.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = torch.optim.SGD(list(g_aux.parameters()) + list(l_head.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        # criterion_g = balsoft_loss
        criterion_l = nn.CrossEntropyLoss()
        criterion_g = nn.CrossEntropyLoss()
        
        # criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_ce = nn.CrossEntropyLoss()

        epoch_loss = []


        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # optimizer_g.zero_grad()
                # optimizer_l.zero_grad()
                optimizer_g_aux.zero_grad()
                # optimizer.zero_grad()

                # backbone更新
                features = net(images, latent_output=True)
                # output_g = g_head(features)
                # loss_g = criterion_g(output_g, labels)
                # loss_g.backward()
                # optimizer_g.step()

                # aux更新
                # output_g_aux = g_aux(features.detach())
                # loss_g_aux = criterion_g(output_g_aux, labels)
                # loss_g_aux.backward()
                # optimizer_g_aux.step()

                # 迁移学习
                outputs_g_aux = g_aux(features.detach())
                # outputs_l_head = l_head(features.detach())
                
                # outputs_g_aux_normalized = F.normalize(outputs_g_aux, dim=1)
                # outputs_l_head_normalized = F.normalize(outputs_l_head, dim=1)

                
                loss = criterion_ce(outputs_g_aux, labels)
                # loss_kl = criterion_kl(nn.functional.log_softmax(outputs_l_head_normalized, dim=1),
                #                     nn.functional.softmax(outputs_g_aux_normalized, dim=1))
                # loss = loss_kl
                loss.backward()
                optimizer_g_aux.step()
                


                # p cls更新
                # output_l = l_head(features.detach())
                # loss_l = criterion_l(output_l, labels)
                # loss_l.backward()
                # optimizer_l.step()

                # loss = loss_l
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), copy.deepcopy(g_aux), copy.deepcopy(l_head), sum(epoch_loss) / len(epoch_loss)
    
    def pfedme_update_weights(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # 专为获取梯度信息设计的local update函数
    def update_weights_gasp_grad(self, net, seed, net_glob, client_id, epoch, gradBag, mu=1, lr=None):
        hookObj,  gradAnalysor = gradBag.get_client(client_id)
        net_glob = net_glob
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)
                hook_handle = logits.register_hook(
                    hookObj.hook_func_tensor)  # hook抓取梯度
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff
                loss.backward()

                if hookObj.has_gradient():
                    # 输入一个batch的梯度和label
                    gradAnalysor.update(
                        hookObj.get_gradient(), labels.cpu().numpy().tolist())
                optimizer.step()
                hook_handle.remove()
                batch_loss.append(loss.item())
                gradAnalysor.print_for_debug()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradBag.load_in(client_id, hookObj, gradAnalysor)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradBag
# 用pid算法进行local update

    def update_weights_GBA_Loss(self, net, seed, epoch, pidloss, mu=1, lr=None):
        
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = pidloss
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_GBA_Finetune(self, net, seed, epoch, pidloss, mu=1, lr=None):
        # 冻结一部分层
        count = 0
        for p in net.parameters():
            if count >= 105:        # 108
                break
            p.requires_grad = False
            count += 1

        filter(lambda p: p.requires_grad, net.parameters())
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = pidloss
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                # print(criterion.pn_diff[70])
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # 用pid算法进行local update
    def update_weights_GBA_Layer(self, net, seed, epoch, GBA_Loss, GBA_Layer, mu=1, lr=None):

        net.train()
        GBA_Layer.train()
        # train and update
        if lr is None:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_sum = [0 for i in range(10)]
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # for i in labels:
                #     label_sum[i] += 1

                # init
                net.zero_grad()
                feat = net(images)
                logits = GBA_Layer(feat)
                loss = GBA_Loss(logits, labels) 
                loss.backward()
                backbone_optimizer.step()
                
                # print(GBA_Loss.pn_diff[52])
                # loss
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print("-----------------------------------------------------------------------")
        return net.state_dict(), GBA_Layer.state_dict(), sum(epoch_loss) / len(epoch_loss)
# global dataset is balanced


def globaltest(net, g_head, test_dataset, args, dataset_class=None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)
            outputs = g_head(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    acc_class_wise = [predict_true_class[i] / total_class_label[i] for i in range(args.num_classes)]
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global

# global dataset is balanced
def globaltest_GBA_Layer(backbone, classifier, test_dataset, args, dataset_class = None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head":0, "middle":0, "tail":0}    #######
    total_3shot = {"head":0, "middle":0, "tail":0} 
    acc_3shot_global = {"head":None, "middle":None, "tail":None}
    backbone.eval()
    classifier.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    distri_class_real = [0 for i in range(100)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            feat = backbone(images)
            outputs = classifier(feat)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # start: cal 3shot metrics
            for label in labels:
                distri_class_real[int(label)] += 1      # 监视真实情况下所有样本的类别分布
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:   # 预测正确且在middle中
                    correct_3shot["middle"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:   # 预测正确且在tail中
                    correct_3shot["tail"] += 1      # 在tail中
            # end 

    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global

def localtest(net, g_head, l_head, test_dataset, dataset_class, idxs, user_id):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(
        test_dataset, idxs), batch_size=args.local_bs, shuffle=False)

    # get overall distribution
    # class_distribution = [0 for _ in range(10000)]  # 10000 >
    # for images, labels in test_loader:
    #     labels = labels.tolist()
    class_distribution_dict = {}

    class_distribution = dataset_class.local_test_distribution[user_id]


    p_mode = 1

    if p_mode == 1:
        # 方案1：
        zero_classes = np.where(class_distribution == 0)[0]
        for i in zero_classes:
            g_head.weight.data[i, :] = -1e10
            l_head.weight.data[i, :] = -1e10
    elif p_mode == 2:
        # 方案2：
        norm = torch.norm(l_head.weight, p=2, dim=1)
        # 将g_head.weight转换为torch.nn.Parameter类型
        g_head.weight = nn.Parameter(g_head.weight * norm.unsqueeze(1))
    elif p_mode == 3:
        # 将class_distribution转换为PyTorch的Tensor
        class_distribution_tensor = torch.from_numpy(class_distribution)
        # 由于g_head.weight的形状是[100, 512]，而class_distribution的形状是[100,]，
        # 所以我们需要将class_distribution扩展为[100, 1]，以便进行元素级别的乘法
        class_distribution_tensor = class_distribution_tensor.view(-1, 1)
        # 将class_distribution_tensor移动到与g_head.weight相同的设备上
        class_distribution_tensor = class_distribution_tensor.to(g_head.weight.device)
        # 进行元素级别的乘法
        g_head.weight = nn.Parameter(g_head.weight * class_distribution_tensor)
    elif p_mode == 4:
        # 将权重和偏置相加，并转换为torch.nn.Parameter
        g_head.weight = nn.Parameter(g_head.weight + l_head.weight)
        g_head.bias = nn.Parameter(g_head.bias + l_head.bias)
    elif p_mode == 5:
        g_head.weight = nn.Parameter(g_head.weight * l_head.weight)
        # g_head.bias = nn.Parameter(g_head.bias * l_head.bias)



    three_shot_dict, _ = shot_split(
        class_distribution, threshold_3shot=[75, 95])
    # three_shot_dict: {"head":[], "middle":[], "tail":[]}   # containg the class id of head, middle and tail respectively
    
    ypred = []
    ytrue = []
    acc_3shot_local = {"head": None, "middle": None, "tail": None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        correct_classwise = [0 for i in range(args.num_classes)]
        total_classwise = [0 for i in range(args.num_classes)]
        acc_classwise = [0 for i in range(args.num_classes)]
        for images, labels in test_loader:
            # inference
            images = images.to(args.device)
            labels = labels.to(args.device)
            features = net(images, latent_output=True)

            if p_mode != 8:
                outputs = g_head(features) + l_head(features)
                # outputs = g_head(features) 
                _, predicted = torch.max(outputs.data, 1)
            else:
                 # use l_head for initial prediction
                l_outputs = l_head(features)
                
                # select top 30% classes
                top_30_percent = int(0.1 * l_outputs.size(1))
                _, top_classes = l_outputs.topk(top_30_percent, dim=1)
                
                # create a mask for selected classes
                mask = torch.zeros_like(l_outputs).scatter_(1, top_classes, 1).bool()
                
                # use g_head for final prediction
                g_outputs = g_head(features)
                
                # apply mask to g_head outputs
                masked_g_outputs = g_outputs.masked_fill(~mask, float('-inf')) 
                
                _, predicted = torch.max(masked_g_outputs.data, 1)

            # calc total metrics
            total += labels.size(0)     # numble of all samples
            # numble of correct predictions
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)
            # f1 = f1_score(y_true=labels,y_pred=predicted)
            # print(f1)
            # all_f1.append(f1)

            # start: cal 3shot metrics
            for label in labels:
                total_classwise[label.cpu().tolist()] += 1 
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correct_classwise[labels[i].cpu().tolist()] += 1 
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end



    ypred = sum(ypred, [])
    ytrue = sum(ytrue, [])
    # print(ypred)
    # print(ytrue)
    f1_macro = f1_score(y_true=ytrue, y_pred=ypred, average='macro')
    f1_weighted = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
    # print(f1)
    # import pdb;pdb.set_trace()
    acc = correct / total

    # start: calc acc_3shot_local
    # acc_3shot_local["head"] = [0, False],False代表无效，平均的时候分母减1
    # 分布不为0，如果没有head，则返回-1，-1不参与平均计算
    acc_3shot_local["head"] = [0, False] if total_3shot["head"] == 0 else [
        (correct_3shot["head"] / total_3shot["head"]), True]
    acc_3shot_local["middle"] = [0, False] if total_3shot["middle"] == 0 else [
        (correct_3shot["middle"] / total_3shot["middle"]), True]
    acc_3shot_local["tail"] = [0, False] if total_3shot["tail"] == 0 else [
        (correct_3shot["tail"] / total_3shot["tail"]), True]
    # end
    for i in range(len(acc_classwise)):
        acc_classwise[i] = correct_classwise[i] / (total_classwise[i]+1e-10)
    # acc = sum(acc_classwise) / len(acc_classwise)
    # print("F1: "+ str(np.mean(f1)))
    return acc, f1_macro, f1_weighted, acc_3shot_local


def localtest_vallina(net, test_dataset, dataset_class, idxs, user_id):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(
        test_dataset, idxs), batch_size=args.local_bs, shuffle=False)


    ypred = []
    ytrue = []
    acc_3shot_local = {"head": None, "middle": None, "tail": None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        correct_classwise = [0 for i in range(args.num_classes)]
        total_classwise = [0 for i in range(args.num_classes)]
        acc_classwise = [0 for i in range(args.num_classes)]
        for images, labels in test_loader:
            # inference
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            # calc total metrics
            total += labels.size(0)     # numble of all samples
            # numble of correct predictions
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)
            # f1 = f1_score(y_true=labels,y_pred=predicted)


    ypred = sum(ypred, [])
    ytrue = sum(ytrue, [])
    # print(ypred)
    # print(ytrue)
    f1_macro = f1_score(y_true=ytrue, y_pred=ypred, average='macro')
    f1_weighted = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
    # print(f1)
    # import pdb;pdb.set_trace()
    acc = correct / total
    return acc, f1_macro, f1_weighted, acc_3shot_local




def calculate_metrics(pred_np, seg_np):
    # pred_np: B,N
    # seg_np: B,N
    b = len(pred_np)
    all_f1 = []
    all_sensitivity = []
    all_specificity = []
    all_ppv = []
    all_npv = []
    for i in range(b):

        f1 = metrics.f1_score(seg_np[i], pred_np[i], average='macro')

        # confusion_matrix = metrics.confusion_matrix(seg_np[i], pred_np[i])  # n_class * n_class(<=17)
        # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)   # n_class，
        # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)   # n_class，
        # TP = np.diag(confusion_matrix)                                  # n_class，
        # TN = confusion_matrix.sum() - (FP + FN + TP)                    # n_class，

        # TPR = []
        # PPV = []
        # for j in range(len(TP)):
        #     if (TP[j] + FN[j]) == 0:
        #         TPR.append(1)
        #     else:
        #         TPR.append(TP[j] / (TP[j] + FN[j]))
        # for j in range(len(TP)):
        #     if (TP[j] + FP[j]) == 0:
        #         PPV.append(1)
        #     else:
        #         PPV.append(TP[j] / (TP[j] + FP[j]))
        # # # Sensitivity, hit rate, recall, or true positive rate
        # # TPR = TP / (TP + FN)
        # # Specificity or true negative rate
        # TNR = TN / (TN + FP)
        # # # Precision or positive predictive value
        # # PPV = TP / (TP + FP)
        # # Negative predictive value
        # NPV = TN / (TN + FN)

        all_f1.append(f1)
        # all_ppv.append(np.mean(PPV))
        # all_npv.append(np.mean(NPV))
        # all_sensitivity.append(np.mean(TPR))
        # all_specificity.append(np.mean(TNR))
    # return all_f1, all_ppv, all_npv, all_sensitivity, all_specificity  # B,
    return all_f1
